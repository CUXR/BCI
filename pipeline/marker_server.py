"""Async WebSocket server that ingests trial markers from Unity.

Counterpart to eeg_to_meta/websocket_server.py: this socket flows the
*other* direction (Unity → Python). The Unity AutoMoveDataCollectionController
publishes one frame per phase transition (cue / start / end) along with
the AutoMove direction string; the marker server normalises these into
canonical pipeline.labels trial types and:

    1. queues them onto an asyncio.Queue for pipeline.collector.py to
       persist into the per-participant trial_log_*.csv,
    2. (optionally) calls a synchronous callback so the BrainFlow
       producer can inject the matching integer marker code into the
       EEG stream — preserving the schema used by
       muse_psychopy_recording_structured.py.

JSON message contract (Unity → Python):

    {
      "type": "marker",
      "phase": "cue" | "start" | "end",
      "direction": "forward" | "backward" | "left_rotate" | "right_rotate",
      "trial_index": 17,
      "run_index": 0,
      "ts_unity": 12.345          // Unity Time.realtimeSinceStartup
    }

For blink trials Unity sends `phase` plus `"trial_type": "blink_intentional"`
in place of `direction`.

Other inbound message types are ignored but logged so we can debug
integration issues without crashing the collector. The server only
emits responses for explicit `{"type": "ping"}` health checks.

Bind to 127.0.0.1 by default — markers contain task metadata that has
no business being on the LAN. Use `--bind 0.0.0.0` only on isolated
networks (matches the security posture set in eeg_to_meta/main.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import websockets
from websockets.server import serve

from . import labels as L

log = logging.getLogger(__name__)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766

VALID_PHASES = ("cue", "start", "end")


# ── Marker record ───────────────────────────────────────────────────


@dataclass
class MarkerEvent:
    """One marker emitted by Unity, normalised for downstream consumers."""

    phase: str                 # "cue" | "start" | "end"
    trial_type: str            # canonical pipeline.labels trial_type
    marker_code: int           # BrainFlow marker code from labels module
    trial_index: int           # 0-based index within the run
    run_index: int             # which AutoMove run this came from
    ts_unity: float            # Unity-side timestamp (Time.realtimeSinceStartup)
    ts_recv: float             # server-side wall clock when received
    raw: dict = field(default_factory=dict, repr=False)  # original JSON

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "trial_type": self.trial_type,
            "marker_code": self.marker_code,
            "trial_index": self.trial_index,
            "run_index": self.run_index,
            "ts_unity": self.ts_unity,
            "ts_recv": self.ts_recv,
        }


# ── Server ──────────────────────────────────────────────────────────


# Type alias for a synchronous BrainFlow injection hook. The collector
# wires this up to BoardShim.insert_marker so the integer code lands on
# the EEG stream at the same instant as the queued event.
MarkerInjector = Callable[[int], None]
AsyncMarkerHook = Callable[[MarkerEvent], Awaitable[None]]


class MarkerServer:
    """WebSocket server that turns raw Unity marker frames into MarkerEvents."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        queue: Optional["asyncio.Queue[MarkerEvent]"] = None,
        on_marker: Optional[MarkerInjector] = None,
        on_marker_async: Optional[AsyncMarkerHook] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.queue: asyncio.Queue[MarkerEvent] = queue or asyncio.Queue()
        self._on_marker = on_marker
        self._on_marker_async = on_marker_async
        self._clients: set[websockets.WebSocketServerProtocol] = set()
        self._server = None

    # ── lifecycle ───────────────────────────────────────────────────

    async def start(self) -> None:
        self._server = await serve(
            self._handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
        )
        log.info("Marker WebSocket listening on ws://%s:%d", self.host, self.port)
        if self.host in ("0.0.0.0", "::"):
            log.warning(
                "Marker WebSocket bound on all interfaces: trial metadata is "
                "reachable from your LAN with no authentication (trusted nets only)."
            )

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            log.info("Marker WebSocket stopped")

    # ── handlers ────────────────────────────────────────────────────

    async def _handler(self, ws: websockets.WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        remote = ws.remote_address
        log.debug("Marker client connected: %s", remote)
        try:
            async for raw in ws:
                await self._on_message(ws, raw)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            log.debug("Marker client disconnected: %s", remote)

    async def _on_message(self, ws, raw: str | bytes) -> None:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            log.warning("Discarding malformed marker frame: %r", raw[:120])
            return

        msg_type = payload.get("type", "marker")
        if msg_type == "ping":
            try:
                await ws.send(json.dumps({"type": "pong", "ts_recv": time.time()}))
            except websockets.ConnectionClosed:
                pass
            return
        if msg_type != "marker":
            log.debug("Ignoring non-marker message type=%r", msg_type)
            return

        event = self._parse_marker(payload)
        if event is None:
            return

        await self.queue.put(event)
        if self._on_marker is not None:
            try:
                self._on_marker(event.marker_code)
            except Exception:                                       # noqa: BLE001
                log.exception("Marker injector raised; continuing")
        if self._on_marker_async is not None:
            try:
                await self._on_marker_async(event)
            except Exception:                                       # noqa: BLE001
                log.exception("Async marker hook raised; continuing")

    # ── parsing ─────────────────────────────────────────────────────

    def _parse_marker(self, payload: dict) -> Optional[MarkerEvent]:
        phase = payload.get("phase")
        if phase not in VALID_PHASES:
            log.warning("Marker missing/invalid phase: %r", payload)
            return None

        trial_type = payload.get("trial_type")
        if trial_type is None:
            direction = payload.get("direction")
            if not isinstance(direction, str):
                log.warning(
                    "Marker missing both trial_type and direction: %r", payload,
                )
                return None
            trial_type = L.UNITY_DIRECTION_TO_TRIAL_TYPE.get(direction)
            if trial_type is None:
                log.warning("Unknown AutoMove direction: %r", direction)
                return None

        if trial_type not in L.TRIAL_TYPE_TO_MARKER_CODE:
            log.warning("Unknown trial_type: %r", trial_type)
            return None

        try:
            trial_index = int(payload.get("trial_index", -1))
            run_index = int(payload.get("run_index", 0))
            ts_unity = float(payload.get("ts_unity", 0.0))
        except (TypeError, ValueError):
            log.warning("Marker has non-numeric index/timestamp fields: %r", payload)
            return None

        return MarkerEvent(
            phase=phase,
            trial_type=trial_type,
            marker_code=L.TRIAL_TYPE_TO_MARKER_CODE[trial_type],
            trial_index=trial_index,
            run_index=run_index,
            ts_unity=ts_unity,
            ts_recv=time.time(),
            raw=payload,
        )

    # ── introspection ───────────────────────────────────────────────

    @property
    def n_clients(self) -> int:
        return len(self._clients)


# ── standalone debug runner ────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bind", default=DEFAULT_HOST,
                   help=f"Listen address (default {DEFAULT_HOST}; use 0.0.0.0 for LAN)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"WebSocket port (default {DEFAULT_PORT})")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="DEBUG-level logging")
    return p


async def _drain_to_stdout(queue: "asyncio.Queue[MarkerEvent]") -> None:
    """Print queued markers in real time — useful for connectivity testing."""
    while True:
        ev = await queue.get()
        print(json.dumps(ev.to_dict()))


async def _amain(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    server = MarkerServer(host=args.bind, port=args.port)
    await server.start()

    drain = asyncio.create_task(_drain_to_stdout(server.queue))

    log.info("Press Ctrl-C to stop.")
    try:
        await asyncio.Future()  # run forever
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        drain.cancel()
        await server.stop()
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_amain()))
    except KeyboardInterrupt:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
