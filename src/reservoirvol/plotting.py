"""Static figures for the report (matplotlib, PNG for the README)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#8a8985"
CRITICAL = "#c4271a"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "figure.dpi": 150,
        "axes.axisbelow": True,
    }
)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def physics(mg, esn, mgc, mc_tab, mc_curve, path):
    from . import reservoir as rc

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    h, train = mgc["horizons"][-1], mgc["train"]
    s = (mg - mg[:train].mean()) / mg[:train].std()
    X = esn.run(s[:, None])
    y = s[h:]
    w, b = rc.ridge_fit(X[200:train], y[200:train], 1e-8)
    seg = slice(train, train + 500)
    a1.plot(y[seg], color=INK, lw=1.8, label="Mackey-Glass (chaotic)")
    a1.plot(X[seg] @ w + b, color=SERIES[0], lw=1.2, ls="--", label=f"ESN forecast made {h} steps earlier")
    a1.set_title("Chaotic benchmark, out of sample", loc="left")
    a1.set_xlabel("time step")
    a1.legend(fontsize=8, loc="upper right")
    a1.set_ylim(top=3.2)
    k = np.arange(1, len(mc_curve["esn"]) + 1)
    a2.plot(k, mc_curve["esn"], color=SERIES[0], label="echo state network")
    a2.plot(k, mc_curve["ring"], color=SERIES[1], label="photonic ring reservoir")
    a2.set_xlabel("delay k (steps)")
    a2.set_ylabel("fraction of u(t - k) recovered")
    a2.set_title("Memory function at radius 0.9", loc="left")
    a2.legend(fontsize=8)
    a3.plot(mc_tab.index, mc_tab["echo state network"], "o-", color=SERIES[0], ms=4)
    a3.plot(mc_tab.index, mc_tab["photonic ring reservoir"], "o-", color=SERIES[1], ms=4)
    a3.axvline(1.0, color=NEUTRAL, ls=":", lw=1)
    a3.set_xlabel("spectral radius / feedback gain")
    a3.set_ylabel("total memory capacity")
    a3.set_title("Memory peaks at the edge of chaos", loc="left")
    _save(fig, path)


def skill(ratio: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(11, 4.0))
    models = [m for m in ratio.index if m != "HAR"]
    x = np.arange(len(models))
    w = 0.26
    for i, c in enumerate(ratio.columns):
        ax.bar(
            x + (i - 1) * w, ratio.loc[models, c], width=w - 0.03, color=SERIES[i], label=c.replace("h = ", "") + "-day"
        )
    ax.axhline(1, color=INK, lw=1)
    ax.set_xticks(
        x,
        [
            m.replace(" (", "\n(")
            .replace("Photonic ring reservoir", "Photonic ring\nreservoir")
            .replace("ESN + HAR-X features", "ESN + HAR-X\nfeatures")
            .replace("Pool: HAR-X + ESN", "Pool:\nHAR-X + ESN")
            .replace("Echo state network", "Echo state\nnetwork")
            for m in models
        ],
        fontsize=8,
    )
    ax.set_ylim(0.7, 1.6)
    ax.set_ylabel("out-of-sample MSE relative to HAR")
    ax.set_title("Volatility forecast errors, 2005-2018 (below the line beats HAR)", loc="left")
    ax.legend(fontsize=8, title="horizon")
    _save(fig, path)


def paths(df, Y, preds, h, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9))
    for ax, (a, b) in zip(axes, [("2008-06-01", "2009-06-30"), ("2017-10-01", "2018-12-31")], strict=True):
        m = (df.index >= a) & (df.index <= b) & np.isfinite(Y[h])
        ix = df.index[m]
        to_vol = lambda z: 100 * np.sqrt(np.exp(z))  # noqa: E731
        ax.plot(ix, to_vol(Y[h][m]), color=INK, lw=1.8, label=f"realised, next {h} days")
        ax.plot(ix, to_vol(preds[h]["HAR"][m]), color=SERIES[1], lw=1.2, label="HAR")
        ax.plot(ix, to_vol(preds[h]["Echo state network"][m]), color=SERIES[0], lw=1.2, label="echo state network")
        ax.set_ylabel("annualised volatility, %")
        ax.set_title(f"{h}-day volatility forecasts, {a[:4]}-{b[:4]}", loc="left")
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].legend(fontsize=8)
    _save(fig, path)


def sensitivity(sens: pd.DataFrame, size_tab: pd.DataFrame, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    labels, pos = [], 0
    for i, kind in enumerate(["esn", "ring"]):
        d = sens.loc[kind]
        for h, row in d.iterrows():
            a1.plot([pos, pos], [row["min"], row["max"]], color=SERIES[i], lw=5, alpha=0.35, solid_capstyle="round")
            a1.plot(pos, row["ensemble"], "o", color=SERIES[i], ms=7, mec=SURFACE, mew=1.5)
            labels.append(f"{'ESN' if kind == 'esn' else 'ring'}\n{h}d")
            pos += 1
    a1.axhline(1, color=INK, lw=1)
    a1.set_xticks(range(pos), labels)
    a1.set_ylabel("MSE relative to HAR")
    a1.set_title("Range over random reservoirs (bar) and their average (dot)", loc="left")
    a2.plot(size_tab.index, size_tab["echo state network"], "o-", color=SERIES[0], label="echo state network")
    a2.plot(size_tab.index, size_tab["photonic ring reservoir"], "o-", color=SERIES[1], label="photonic ring reservoir")
    a2.axhline(1, color=INK, lw=1)
    a2.set_xscale("log")
    a2.set_xticks(list(size_tab.index), [str(n) for n in size_tab.index])
    a2.minorticks_off()
    a2.set_xlabel("number of nodes")
    a2.set_ylabel("MSE relative to HAR, 1 day")
    a2.set_title("How large does the reservoir need to be?", loc="left")
    a2.legend(fontsize=8)
    _save(fig, path)
