"""Command-line entry point: ``python -m reservoirvol run [--config ...] [--out ...] [--fast]``."""

from __future__ import annotations

import argparse

from .volatility import ROOT


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="reservoirvol", description="Reservoir computing for volatility forecasting.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline and write reports/")
    r.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    r.add_argument("--out", default=str(ROOT / "reports"))
    r.add_argument("--fast", action="store_true", help="small reservoirs and grids (for CI)")
    args = ap.parse_args(argv)
    from .pipeline import run

    m = run(args.config, args.out, fast=args.fast)
    bad = [c for c in m["checks"] if c["status"] == "FAIL"]
    print(f"Reports written to {args.out} ({len(m['checks'])} checks, {len(bad)} FAIL)")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
