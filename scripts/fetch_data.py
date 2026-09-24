"""Rebuild data/raw: S&P 500 daily OHLC bundled with the ``arch`` package (``pip install arch``) and the
CBOE VIX from datahub.io.

Usage:  python scripts/fetch_data.py [--out data/raw]
"""

from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import pandas as pd

VIX = "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"


def main() -> None:
    import arch.data.sp500

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    px = arch.data.sp500.load()[["Open", "High", "Low", "Close"]]
    px.index.name = "date"
    px.to_csv(out / "sp500_ohlc_daily.csv", float_format="%.4f")
    with urllib.request.urlopen(VIX, timeout=60) as r:  # noqa: S310 (fixed https URL)
        v = pd.read_csv(io.BytesIO(r.read()), parse_dates=["DATE"])
    v = v.rename(columns={"DATE": "date", "CLOSE": "vix"})[["date", "vix"]]
    v = v[(v["date"] >= px.index[0] - pd.Timedelta(days=40)) & (v["date"] <= px.index[-1] + pd.Timedelta(days=31))]
    v.to_csv(out / "vix_daily.csv", index=False, float_format="%.2f")
    print(f"sp500_ohlc_daily.csv {len(px)} rows, vix_daily.csv {len(v)} rows")


if __name__ == "__main__":
    main()
