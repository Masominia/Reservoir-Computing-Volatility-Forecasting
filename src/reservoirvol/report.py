"""Small helpers to write Markdown tables without extra dependencies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fmt(x, digits: int) -> str:
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{x:,d}"
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "n/a"
        if x == 0:
            x = 0.0  # normalise -0.0
        if x != 0 and (abs(x) < 0.01 or abs(x) >= 1e7):
            return f"{x:.2e}"
        return f"{x:,.{digits}f}"
    return str(x)


def md_table(df: pd.DataFrame, digits: int = 2, index: bool = True) -> str:
    if index:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join(str(x) for x in c if str(x)) for c in df.columns]
        names = [n or " " * i for i, n in enumerate(df.index.names)]
        d = df.reset_index(names=names if len(names) > 1 else names[0])
    else:
        d = df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in d.itertuples(index=False):  # keeps per-column dtypes (ints stay ints)
        lines.append("| " + " | ".join(_fmt(v, digits) for v in row) + " |")
    return "\n".join(lines)
