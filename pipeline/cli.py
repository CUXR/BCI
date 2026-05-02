"""Single CLI for the final-realtime BCI pipeline.

Subcommands:

    collect       record an AutoMove session for one participant
                  (delegates to pipeline.collector)
    personalize   refit MI + blink models on one participant's data
                  (delegates to pipeline.personalize)
    train         train the base / cross-subject model
                  (delegates to ml_pipeline/train.py)
    realtime      personalize (if needed) and launch the EEG → Unity
                  inference loop, applying per-class thresholds

Run `python -m pipeline <subcommand> --help` for subcommand-specific
options. The `pipeline` package also exposes a `__main__` shim so
`python -m pipeline` is equivalent to `python -m pipeline.cli`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger("pipeline.cli")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ML_DIR = _PROJECT_ROOT / "ml_pipeline"
_EEG_DIR = _PROJECT_ROOT / "eeg_to_meta"
_DEFAULT_DATA_ROOT = _PROJECT_ROOT / "data"
_DEFAULT_PARTICIPANTS_DIR = _ML_DIR / "models" / "participants"


# ── Subcommand implementations ─────────────────────────────────────


def _missing_dep_msg(exc: ModuleNotFoundError) -> str:
    name = exc.name or "?"
    return (
        f"Missing dependency '{name}'. Install pipeline requirements with:\n"
        f"  pip install -r eeg_to_meta/requirements.txt\n"
        f"  pip install -r ml_pipeline/requirements.txt"
    )


def _do_collect(rest_argv: list[str]) -> int:
    try:
        from .collector import _amain as collector_amain
    except ModuleNotFoundError as exc:
        log.error("%s", _missing_dep_msg(exc))
        return 4
    try:
        return asyncio.run(collector_amain(rest_argv))
    except KeyboardInterrupt:
        return 0


def _do_personalize(rest_argv: list[str]) -> int:
    try:
        from .personalize import main as personalize_main
    except ModuleNotFoundError as exc:
        log.error("%s", _missing_dep_msg(exc))
        return 4
    return personalize_main(rest_argv)


def _do_train(rest_argv: list[str]) -> int:
    """Run the base trainer in ml_pipeline/.

    The legacy trainer ignores extra args, but we forward any flags via
    sys.argv mutation in case a future 4-class trainer wants its own CLI.
    """
    if str(_ML_DIR) not in sys.path:
        sys.path.insert(0, str(_ML_DIR))
    import train as ml_train  # type: ignore[import-not-found]   # noqa: PLC0415
    saved_argv = sys.argv
    sys.argv = ["train"] + list(rest_argv)
    try:
        ml_train.main()
    finally:
        sys.argv = saved_argv
    return 0


def _do_realtime(rest_argv: list[str]) -> int:
    args = _realtime_argparser().parse_args(rest_argv)
    _configure_logging(args.verbose)

    bundle_path: Path
    if args.model:
        bundle_path = Path(args.model)
        if not bundle_path.exists():
            log.error("Model not found: %s", bundle_path)
            return 2
    else:
        bundle_path = _DEFAULT_PARTICIPANTS_DIR / f"sub{args.participant:02d}.pkl"

    needs_personalize = (
        args.re_personalize
        or (not bundle_path.exists() and not args.skip_personalize)
    )
    if needs_personalize:
        log.info("Personalising model for sub%02d ...", args.participant)
        try:
            from .personalize import personalize
        except ModuleNotFoundError as exc:
            log.error("%s", _missing_dep_msg(exc))
            return 4
        try:
            bundle_path = personalize(
                participant=args.participant,
                data_root=args.data_root,
                output_dir=bundle_path.parent,
            )
        except FileNotFoundError as exc:
            log.error("%s", exc)
            return 2
        except RuntimeError as exc:
            log.error("Personalisation failed: %s", exc)
            return 3
    elif not bundle_path.exists():
        log.error(
            "Bundle %s does not exist and --skip-personalize was passed; "
            "run `python -m pipeline personalize --participant %d` first.",
            bundle_path, args.participant,
        )
        return 2

    log.info("Using model bundle: %s", bundle_path)
    return _launch_eeg_to_meta(args, bundle_path)


# ── Realtime helpers ───────────────────────────────────────────────


def _realtime_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline realtime",
        description="Personalise (if needed) and stream EEG predictions to Unity.",
    )
    p.add_argument("--participant", type=int, required=True,
                   help="Participant number used to locate the personalised bundle")
    p.add_argument("--re-personalize", action="store_true",
                   help="Force a fresh personalisation pass before launching realtime")
    p.add_argument("--skip-personalize", action="store_true",
                   help="Never run personalisation; fail if bundle is missing")
    p.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT,
                   help=f"Root of per-participant data (default: {_DEFAULT_DATA_ROOT})")
    p.add_argument("--model", type=Path, default=None,
                   help="Explicit .pkl bundle path (overrides the auto-resolved one)")
    p.add_argument("--mi-threshold", type=float, default=None,
                   help="Override per-class MI threshold (applied to all 4 MI classes)")
    p.add_argument("--blink-threshold", type=float, default=None,
                   help="Override blink confidence threshold")
    # Forwarded straight to eeg_to_meta/main.py
    p.add_argument("--mock", action="store_true",
                   help="Use synthetic in-process EEG (no Muse needed)")
    p.add_argument("--simulate", action="store_true",
                   help="Use BrainFlow's synthetic board")
    p.add_argument("--serial", type=str, default=None,
                   help="Muse BLE serial number (e.g. Muse-15C3)")
    p.add_argument("--no-ws", action="store_true",
                   help="Disable the prediction WebSocket server")
    p.add_argument("--ws-host", type=str, default=None,
                   help="WebSocket bind address (default from eeg_to_meta config)")
    p.add_argument("--ws-port", type=int, default=None,
                   help="WebSocket port (default from eeg_to_meta config)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="DEBUG-level logging")
    return p


def _launch_eeg_to_meta(args: argparse.Namespace, bundle_path: Path) -> int:
    """Run eeg_to_meta/main.py:run_pipeline in-process so we can mutate config."""
    if str(_EEG_DIR) not in sys.path:
        sys.path.insert(0, str(_EEG_DIR))

    try:
        import config as eeg_config                                 # noqa: PLC0415
        import main as eeg_main                                     # noqa: PLC0415
    except ModuleNotFoundError as exc:
        log.error("%s", _missing_dep_msg(exc))
        return 4

    if args.mi_threshold is not None:
        eeg_config.MI_CONFIDENCE_THRESHOLD = float(args.mi_threshold)
        for k in list(eeg_config.CLASS_THRESHOLDS):
            if k.startswith("mi_"):
                eeg_config.CLASS_THRESHOLDS[k] = float(args.mi_threshold)
        log.info("Override MI threshold → %.2f", args.mi_threshold)
    if args.blink_threshold is not None:
        eeg_config.BLINK_CONFIDENCE_THRESHOLD = float(args.blink_threshold)
        eeg_config.CLASS_THRESHOLDS["intentional_blink"] = float(args.blink_threshold)
        log.info("Override blink threshold → %.2f", args.blink_threshold)

    eeg_args = argparse.Namespace(
        mock=args.mock,
        simulate=args.simulate,
        serial=args.serial,
        model=str(bundle_path),
        no_ws=args.no_ws,
        ws_host=args.ws_host if args.ws_host is not None else eeg_config.WS_HOST,
        ws_port=args.ws_port if args.ws_port is not None else eeg_config.WS_PORT,
    )

    log.info(
        "Launching eeg_to_meta (mock=%s, simulate=%s, ws=%s:%d)",
        eeg_args.mock, eeg_args.simulate,
        eeg_args.ws_host, eeg_args.ws_port,
    )
    try:
        asyncio.run(eeg_main.run_pipeline(eeg_args))
    except KeyboardInterrupt:
        return 0
    return 0


# ── Top-level argparse / dispatch ───────────────────────────────────


SUBCOMMANDS = {
    "collect": (_do_collect, "Record AutoMove session(s) for one participant"),
    "personalize": (_do_personalize, "Refit MI + blink models on one participant's data"),
    "train": (_do_train, "Train the base / cross-subject model (ml_pipeline/train.py)"),
    "realtime": (_do_realtime, "Personalise (if needed) and stream EEG predictions to Unity"),
}


def _print_top_help() -> None:
    print("Usage: python -m pipeline <command> [...]\n")
    print("Commands:")
    width = max(len(c) for c in SUBCOMMANDS)
    for name, (_, desc) in SUBCOMMANDS.items():
        print(f"  {name:<{width}}   {desc}")
    print(
        "\nUse `python -m pipeline <command> --help` for command-specific options."
    )


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    if not argv or argv[0] in ("-h", "--help"):
        _print_top_help()
        return 0 if argv else 1

    cmd, rest = argv[0], argv[1:]
    if cmd not in SUBCOMMANDS:
        print(f"Unknown command: {cmd}\n", file=sys.stderr)
        _print_top_help()
        return 2

    handler, _ = SUBCOMMANDS[cmd]
    try:
        return int(handler(rest) or 0)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
