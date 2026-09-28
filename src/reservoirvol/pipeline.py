"""End-to-end run: physics checks of the reservoirs (chaotic benchmark, memory capacity) -> volatility data ->
hyperparameter selection on pre-test data -> walk-forward forecasts 2005-2018 for all models -> scores and
tests -> seed and size sensitivity -> report and validation checks."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import plotting
from . import reservoir as rc
from . import volatility as vol
from .report import md_table

ROOT = vol.ROOT


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def make(kind: str, n: int, params: dict, seed: int):
    cls = rc.EchoStateNetwork if kind == "esn" else rc.RingReservoir
    return cls(n=n, seed=seed, **params)


def select(kind, grid, U, Y, index, cfg, seed) -> dict:
    """Best (reservoir parameters, ridge) per horizon: fit before the validation years, score on them."""
    rcfg = cfg["reservoir"]
    val = np.isin(index.year, cfg["validation_years"])
    v0 = int(np.flatnonzero(val)[0])
    end = int(np.flatnonzero(val)[-1]) + 1
    keys = list(grid)
    best = {h: (np.inf, None, None) for h in Y}
    for combo in itertools.product(*grid.values()):
        params = dict(zip(keys, combo, strict=True))
        states = [
            np.column_stack([make(kind, rcfg["nodes"], params, seed + s).run(U[:end]), U[:end]])
            for s in range(rcfg["selection_seeds"])
        ]
        for h, y in Y.items():
            tr = np.arange(rcfg["washout"], v0 - h)
            te = np.arange(v0, end)
            te = te[np.isfinite(y[te])]
            for lam in rcfg["ridge"]:
                p = np.mean([F[te] @ w + b for F in states for w, b in [rc.ridge_fit(F[tr], y[tr], lam)]], axis=0)
                mse = float(np.mean((p - y[te]) ** 2))
                if mse < best[h][0]:
                    best[h] = (mse, params, lam)
    return best


def run(cfg_path=ROOT / "configs" / "default.yaml", out=ROOT / "reports", fast: bool = False) -> dict:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    rcfg, ph = cfg["reservoir"], cfg["physics"]
    if fast:
        rcfg.update(nodes=60, ensemble=2, selection_seeds=1, ridge=[0.01, 0.1])
        rcfg["esn_grid"] = {"spectral_radius": [0.8], "leak": [0.2, 1.0], "input_scale": [0.3]}
        rcfg["ring_grid"] = {"feedback": [0.8], "input_scale": [0.2], "phase": [0.0]}
        ph["memory"]["radii"] = [0.4, 0.9, 1.3]
        cfg["sensitivity"]["sizes"] = [25, 60]
    out = Path(out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    md, metrics, checks = [], {}, []

    def check(area, name, value, threshold, ok, warn_only=False):
        checks.append(
            {
                "area": area,
                "check": name,
                "value": value,
                "threshold": threshold,
                "status": "PASS" if ok else ("WARN" if warn_only else "FAIL"),
            }
        )

    # 1. Physics checks
    mgc = ph["mackey_glass"]
    mg = rc.mackey_glass(mgc["length"])
    esn_p = rc.EchoStateNetwork(ph["nodes"], 0.9, 0.5, 0.5, seed=seed)
    ring_p = rc.RingReservoir(ph["nodes"], 0.8, 0.3, 0.2, seed=seed)
    mg_tab = pd.DataFrame(
        {
            "linear AR(50)": [rc.ar_forecast_nrmse(mg, h, mgc["train"]) for h in mgc["horizons"]],
            "echo state network": [rc.direct_forecast_nrmse(esn_p, mg, h, mgc["train"]) for h in mgc["horizons"]],
            "photonic ring reservoir": [rc.direct_forecast_nrmse(ring_p, mg, h, mgc["train"]) for h in mgc["horizons"]],
        },
        index=[f"{h} step{'s' if h > 1 else ''} ahead" for h in mgc["horizons"]],
    )
    mm = ph["memory"]
    mc_rows = []
    for r in mm["radii"]:
        mc_rows.append(
            {
                "spectral radius / feedback": r,
                "echo state network": rc.memory_capacity(
                    rc.EchoStateNetwork(mm["nodes"], r, 1.0, 0.1, seed=seed), mm["k_max"]
                ).sum(),
                "photonic ring reservoir": rc.memory_capacity(
                    rc.RingReservoir(mm["nodes"], r, 0.1, 0.0, seed=seed), mm["k_max"]
                ).sum(),
            }
        )
    mc_tab = pd.DataFrame(mc_rows).set_index("spectral radius / feedback")
    mc_curve = {
        "esn": rc.memory_capacity(rc.EchoStateNetwork(mm["nodes"], 0.9, 1.0, 0.1, seed=seed), mm["k_max"]),
        "ring": rc.memory_capacity(rc.RingReservoir(mm["nodes"], 0.9, 0.1, 0.0, seed=seed), mm["k_max"]),
    }
    metrics["physics"] = {"mackey_glass_nrmse": mg_tab.to_dict(), "memory_capacity": mc_tab.to_dict()}
    hl = mg_tab.index[-1]
    check(
        "Physics",
        f"Mackey-Glass, {hl}: ESN error relative to the linear model",
        float(mg_tab.loc[hl, "echo state network"] / mg_tab.loc[hl, "linear AR(50)"]),
        "< 0.5",
        mg_tab.loc[hl, "echo state network"] < 0.5 * mg_tab.loc[hl, "linear AR(50)"],
    )
    check(
        "Physics",
        "memory capacity never exceeds the number of nodes (Jaeger bound)",
        float(mc_tab.max().max()),
        f"<= {mm['nodes']}",
        mc_tab.max().max() <= mm["nodes"],
    )
    peak, beyond = mc_tab.loc[[r for r in mm["radii"] if r <= 1.0]].max(), mc_tab.loc[max(mm["radii"])]
    check(
        "Physics",
        "memory falls beyond the edge of chaos, both reservoirs (MC at the largest gain / peak MC)",
        float((beyond / peak).max()),
        "< 0.8",
        (beyond / peak).max() < 0.8,
    )
    check(
        "Physics",
        "ring reservoir: memory collapses once the feedback gain exceeds 1",
        float(beyond["photonic ring reservoir"] / peak["photonic ring reservoir"]),
        "< 0.1",
        beyond["photonic ring reservoir"] < 0.1 * peak["photonic ring reservoir"],
    )
    plotting.physics(mg, esn_p, mgc, mc_tab, mc_curve, figs / "physics_checks.png")
    md += [
        "# ReservoirVol report",
        "",
        f"Seed {seed}.",
        "",
        "## 1. Do the reservoirs work as physics says they should?",
        "",
        f"Mackey-Glass chaotic series (tau = 17), {ph['nodes']} nodes, readout trained on {mgc['train']:,} points; "
        "normalised RMSE on the following points:",
        "",
        md_table(mg_tab, 4),
        "",
        f"Linear memory capacity ({mm['nodes']} nodes, theoretical maximum {mm['nodes']}):",
        "",
        md_table(mc_tab, 1),
        "",
    ]

    # 2. Data
    df = vol.load()
    H = cfg["horizons"]
    Yd = vol.targets(df, H)
    Y = {h: Yd[h].to_numpy() for h in H}
    first = cfg["first_test_year"]
    fit_end = int(np.searchsorted(df.index, pd.Timestamp(f"{first}-01-01")))
    U = vol.reservoir_inputs(df, fit_end)
    r = df["ret"].to_numpy()
    ac = [float(df["logv"].autocorr(k)) for k in (1, 5, 22, 66)]
    metrics["data"] = {
        "days": len(df),
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "gk_vol_pct": float(100 * np.sqrt(df["gk"].mean())),
        "close_to_close_vol_pct": float(100 * r.std() * np.sqrt(vol.ANN)),
        "logv_autocorr": ac,
    }
    md += [
        "## 2. Data and target",
        "",
        f"S&P 500, {df.index[0].date()} to {df.index[-1].date()} ({len(df):,} days). Target: log of the average "
        "Garman-Klass variance over the next h days. Average Garman-Klass volatility "
        f"{metrics['data']['gk_vol_pct']:.1f}% (open-to-close) against "
        f"{metrics['data']['close_to_close_vol_pct']:.1f}% close-to-close. "
        "Autocorrelation of daily log variance at lags 1, 5, 22, 66: "
        f"{', '.join(f'{a:.2f}' for a in ac)} (long memory).",
        "",
    ]

    # 3. Hyperparameters (pre-test data only)
    sel = {
        "esn": select("esn", rcfg["esn_grid"], U, Y, df.index, cfg, seed),
        "ring": select("ring", rcfg["ring_grid"], U, Y, df.index, cfg, seed),
    }
    sel_rows = []
    for kind, lab in (("esn", "echo state network"), ("ring", "photonic ring reservoir")):
        for h in H:
            mse, params, lam = sel[kind][h]
            sel_rows.append(
                {
                    "reservoir": lab,
                    "horizon": h,
                    "selected": ", ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in params.items()),
                    "ridge": f"{lam:g}",
                    "validation MSE": mse,
                }
            )
    sel_tab = pd.DataFrame(sel_rows).set_index(["reservoir", "horizon"])
    metrics["selection"] = {f"{a} h={b}": v for (a, b), v in sel_tab.to_dict(orient="index").items()}
    md += [
        "## 3. Hyperparameters, chosen before the test period",
        "",
        f"Grid search with readouts fitted on data before {cfg['validation_years'][0]} and scored on "
        f"{cfg['validation_years'][0]}-{cfg['validation_years'][-1]}; nothing from {first} onward is used. "
        f"{rcfg['nodes']} nodes.",
        "",
        md_table(sel_tab, 3),
        "",
    ]

    # 4. Walk-forward forecasts
    preds, svars, singles = {}, {}, {}
    ew = vol.ewma_feature(r)[:, None]
    hx = vol.har_features(df, True)
    for h in H:
        y = Y[h]
        P = {}
        rw = df["logv"].rolling(h).mean().to_numpy()
        P["Random walk"] = (rw, np.full(len(y), np.nanvar(y[:fit_end] - rw[:fit_end])))
        P["EWMA (RiskMetrics)"] = vol.walk_forward(ew, y, df.index, h, first)
        P["GARCH(1,1)"] = vol.garch_walk_forward(df, y, h, first)
        P["HAR"] = vol.walk_forward(vol.har_features(df), y, df.index, h, first)
        P["HAR-X (VIX, leverage)"] = vol.walk_forward(hx, y, df.index, h, first)
        for kind, lab in (("esn", "Echo state network"), ("ring", "Photonic ring reservoir")):
            _, params, lam = sel[kind][h]
            runs = []
            for s in range(rcfg["ensemble"]):
                F = np.column_stack([make(kind, rcfg["nodes"], params, seed + 100 + s).run(U), U])
                runs.append(vol.walk_forward(F, y, df.index, h, first, lam, rcfg["washout"]))
                if kind == "esn" and s == 0:
                    Fh = np.column_stack([F, np.nan_to_num(hx)])
                    P["ESN + HAR-X features"] = vol.walk_forward(Fh, y, df.index, h, first, lam, rcfg["washout"])
            P[lab] = (np.mean([a for a, _ in runs], axis=0), np.mean([b for _, b in runs], axis=0))
            singles[(kind, h)] = [a for a, _ in runs]
        a, b = P["HAR-X (VIX, leverage)"], P["Echo state network"]
        P["Pool: HAR-X + ESN"] = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
        preds[h] = {k: v[0] for k, v in P.items()}
        svars[h] = {k: v[1] for k, v in P.items()}

    order = list(preds[H[0]])
    score, tests = {}, []
    for h in H:
        y = Y[h]
        te = (df.index.year >= first) & np.isfinite(y)
        e2 = {k: (p[te] - y[te]) ** 2 for k, p in preds[h].items()}
        ql = {k: vol.qlike(np.exp(y[te]), np.exp(preds[h][k][te] + 0.5 * svars[h][k][te])) for k in order}
        for k in order:
            score[(k, h)] = {
                "MSE (log variance)": e2[k].mean(),
                "MSE / HAR": e2[k].mean() / e2["HAR"].mean(),
                "R2 out of sample": 1 - e2[k].mean() / np.var(y[te]),
                "QLIKE": ql[k].mean(),
                "QLIKE / HAR": ql[k].mean() / ql["HAR"].mean(),
            }
        for k in ("Echo state network", "Photonic ring reservoir", "ESN + HAR-X features", "Pool: HAR-X + ESN"):
            for bench in ("HAR", "HAR-X (VIX, leverage)"):
                s1, p1 = vol.dm(e2[k] - e2[bench], h)
                s2, p2 = vol.dm(ql[k] - ql[bench], h)
                tests.append(
                    {
                        "model": k,
                        "vs": bench,
                        "horizon": h,
                        "DM (MSE)": s1,
                        "p (MSE)": p1,
                        "DM (QLIKE)": s2,
                        "p (QLIKE)": p2,
                    }
                )
    S = pd.DataFrame(score).T
    S.index.names = ["model", "horizon"]
    ratio = S["MSE / HAR"].unstack("horizon").loc[order]
    qratio = S["QLIKE / HAR"].unstack("horizon").loc[order]
    r2 = S["R2 out of sample"].unstack("horizon").loc[order]
    ratio.columns = qratio.columns = r2.columns = [f"h = {h}" for h in H]
    T = pd.DataFrame(tests).set_index(["model", "vs", "horizon"])
    n_test = int(((df.index.year >= first) & np.isfinite(Y[1])).sum())
    metrics["oos"] = {
        "n_days": n_test,
        "mse_ratio": ratio.to_dict(),
        "qlike_ratio": qratio.to_dict(),
        "r2": r2.to_dict(),
        "dm": {" | ".join(map(str, k)): v for k, v in T.to_dict(orient="index").items()},
    }
    esn1 = ratio.loc["Echo state network", "h = 1"]
    check("Forecast", "ESN beats HAR at one day (MSE ratio)", float(esn1), "< 1", esn1 < 1)
    check(
        "Forecast",
        "HAR beats the random walk at every horizon (sanity of the benchmark)",
        float(ratio.loc["Random walk"].min()),
        "> 1",
        ratio.loc["Random walk"].min() > 1,
    )
    pool_worst = float((ratio.loc["Pool: HAR-X + ESN"] / ratio.loc["HAR-X (VIX, leverage)"]).max())
    check(
        "Forecast",
        "pool of HAR-X and ESN is never worse than HAR-X by more than 3% (MSE)",
        pool_worst,
        "< 1.03",
        pool_worst < 1.03,
        warn_only=True,
    )

    # Sub-periods
    sub = {}
    for lab, (a, b) in {
        "2005-2007": (2005, 2007),
        "2008-2009 (crisis)": (2008, 2009),
        "2010-2018": (2010, 2018),
    }.items():
        for h in H:
            y = Y[h]
            m = (df.index.year >= a) & (df.index.year <= b) & np.isfinite(y)
            base = np.mean((preds[h]["HAR-X (VIX, leverage)"][m] - y[m]) ** 2)
            sub[(lab, f"h = {h}")] = np.mean((preds[h]["Echo state network"][m] - y[m]) ** 2) / base
    sub = pd.Series(sub).unstack()[[f"h = {h}" for h in H]]
    metrics["oos"]["esn_vs_harx_by_period"] = sub.to_dict()
    plotting.skill(ratio, figs / "oos_skill.png")
    plotting.paths(df, Y, preds, 5 if 5 in H else H[0], figs / "forecast_paths.png")
    md += [
        f"## 4. Out-of-sample forecasts, {first}-2018",
        "",
        f"{n_test:,} forecast days; all readouts re-estimated each January on an expanding window; reservoir "
        f"forecasts average {rcfg['ensemble']} random reservoirs.",
        "",
        "### MSE of log variance relative to HAR (below 1 = better)",
        "",
        md_table(ratio, 3),
        "",
        "### QLIKE relative to HAR",
        "",
        md_table(qratio, 3),
        "",
        "### Out-of-sample R2",
        "",
        md_table(r2, 3),
        "",
        "### Diebold-Mariano tests (negative = the model in the first column is better)",
        "",
        md_table(T, 3),
        "",
        "### ESN relative to HAR-X by period (MSE ratio)",
        "",
        md_table(sub, 3),
        "",
    ]

    # 5. Seed and size sensitivity
    sens_rows = []
    for (kind, h), runs in singles.items():
        y = Y[h]
        te = (df.index.year >= first) & np.isfinite(y)
        base = np.mean((preds[h]["HAR"][te] - y[te]) ** 2)
        one = np.array([np.mean((p[te] - y[te]) ** 2) / base for p in runs])
        ens = np.mean((np.mean(runs, axis=0)[te] - y[te]) ** 2) / base
        sens_rows.append(
            {
                "reservoir": kind,
                "horizon": h,
                "single reservoirs: mean": one.mean(),
                "min": one.min(),
                "max": one.max(),
                "ensemble": ens,
            }
        )
    sens = pd.DataFrame(sens_rows).set_index(["reservoir", "horizon"])
    sc = cfg["sensitivity"]
    hs = sc["horizon"]
    y = Y[hs]
    te = (df.index.year >= first) & np.isfinite(y)
    base = np.mean((preds[hs]["HAR"][te] - y[te]) ** 2)
    size_rows = []
    for n in sc["sizes"]:
        row = {"nodes": n}
        for kind, lab in (("esn", "echo state network"), ("ring", "photonic ring reservoir")):
            _, params, lam = sel[kind][hs]
            ps = [
                vol.walk_forward(
                    np.column_stack([make(kind, n, params, seed + 500 + s).run(U), U]),
                    y,
                    df.index,
                    hs,
                    first,
                    lam,
                    rcfg["washout"],
                )[0]
                for s in range(min(3, rcfg["ensemble"]))
            ]
            row[lab] = np.mean((np.mean(ps, axis=0)[te] - y[te]) ** 2) / base
        size_rows.append(row)
    size_tab = pd.DataFrame(size_rows).set_index("nodes")
    metrics["sensitivity"] = {
        "seeds": {f"{a} h={b}": v for (a, b), v in sens.to_dict(orient="index").items()},
        "size": size_tab.to_dict(),
    }
    spread = float((sens["max"] - sens["min"]).max())
    check(
        "Robustness",
        "largest spread of MSE ratio across random reservoirs",
        spread,
        "< 0.10",
        spread < 0.10,
        warn_only=True,
    )
    plotting.sensitivity(sens, size_tab, figs / "sensitivity.png")
    md += [
        "## 5. Does the random reservoir matter?",
        "",
        "MSE relative to HAR for each of the random reservoirs on its own, and for their average:",
        "",
        md_table(sens, 3),
        "",
        f"Reservoir size (h = {hs}, average of 3 reservoirs, same hyperparameters):",
        "",
        md_table(size_tab, 3),
        "",
    ]

    ch = pd.DataFrame(checks)
    val = [
        "# Validation checks",
        "",
        md_table(ch, 4, index=False),
        "",
        f"{(ch.status == 'PASS').sum()} PASS, {(ch.status == 'WARN').sum()} WARN, {(ch.status == 'FAIL').sum()} FAIL.",
        "",
    ]
    metrics["checks"] = checks
    (out / "report.md").write_text("\n".join(md))
    (out / "validation.md").write_text("\n".join(val))
    (out / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2))
    pd.DataFrame({f"h{h} {k}": p for h in H for k, p in preds[h].items()}, index=df.index).loc[str(first) :].round(
        4
    ).to_csv(out / "forecasts.csv")
    return metrics
