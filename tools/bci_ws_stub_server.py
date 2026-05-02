"""
Stub WebSocket server for manually testing BCICommandClient.

Sends one BCI-style message per keypress. Run with:

    uv run --with websockets python tools/bci_ws_stub_server.py

Then in this server's terminal press w/a/s/d (followed by Enter) to emit
FORWARD / LEFT / BACKWARD / RIGHT, or q to quit.
"""
import asyncio
import json
import sys
import time

import websockets

KEY_TO_LABEL = {
    "w": "FORWARD",
    "a": "LEFT",
    "s": "BACKWARD",
    "d": "RIGHT",
}

clients: set = set()


async def handler(ws):
    clients.add(ws)
    print(f"Client connected ({len(clients)} total)")
    try:
        async for _ in ws:
            pass
    finally:
        clients.discard(ws)
        print(f"Client disconnected ({len(clients)} total)")


async def stdin_loop():
    loop = asyncio.get_event_loop()
    print("Keys: w=FORWARD a=LEFT s=BACKWARD d=RIGHT q=quit")
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        ch = line.strip().lower()
        if ch == "q":
            break
        if ch not in KEY_TO_LABEL:
            continue
        msg = json.dumps({
            "label": KEY_TO_LABEL[ch],
            "confidence": 0.9,
            "ts": time.time(),
        })
        for c in list(clients):
            try:
                await c.send(msg)
            except Exception:
                pass
        print(f"sent {msg}")


async def main():
    async with websockets.serve(handler, "localhost", 5000):
        print("WS stub listening on ws://localhost:5000")
        await stdin_loop()


if __name__ == "__main__":
    asyncio.run(main())
