import numpy as np
import pandas as pd
import pytest

from reservoirvol import volatility as vol


def test_data_and_garman_klass():
    df = vol.load()
    assert len(df) == 5030 and df.index[0].year == 1999 and df.index[-1].year == 2018
    assert (df["gk"] > 0).all() and np.isfinite(df).all().all()
    # Garman-Klass on a simulated driftless diffusion recovers the variance
    rng = np.random.default_rng(0)
    steps = (
        rng.standard_normal((2000, 4000)) * 0.01 / np.sqrt(4000)
    )  # fine grid: discrete sampling biases the range down
    path = np.cumsum(steps, axis=1)
    hi, lo, c = np.maximum(path.max(1), 0), np.minimum(path.min(1), 0), path[:, -1]
    gk = 0.5 * (hi - lo) ** 2 - (2 * np.log(2) - 1) * c**2
    assert gk.mean() == pytest.approx(1e-4, rel=0.08)
    assert gk.std() < 0.5 * (c**2).std()  # far less noisy than squared returns


def test_targets_use_future_only():
    df = vol.load().iloc[:300]
    y = vol.targets(df, [5])[5]
    assert y.iloc[10] == pytest.approx(np.log(df["gk"].iloc[11:16].mean()))
    assert y.iloc[-5:].isna().all()


def test_walk_forward_has_no_look_ahead():
    idx = pd.bdate_range("2000-01-03", periods=1500)
    rng = np.random.default_rng(1)
    F = rng.standard_normal((1500, 2))
    y = F @ np.array([1.0, -1.0]) + 0.1 * rng.standard_normal(1500)
    pred, svar = vol.walk_forward(F, y, idx, 5, 2003)
    first = int(np.flatnonzero(idx.year == 2003)[0])
    assert np.isnan(pred[:first]).all() and np.isfinite(pred[first:]).all()
    y2 = y.copy()
    y2[first - 5 :] += 100.0  # changing targets not yet observable at the first refit ...
    pred2, _ = vol.walk_forward(F, y2, idx, 5, 2003)
    year1 = idx.year == 2003
    assert np.allclose(pred[year1], pred2[year1])  # ... leaves the first year's forecasts unchanged
    assert np.mean((pred[first:] - y[first:]) ** 2) < 0.02 and svar[first] == pytest.approx(0.01, rel=0.3)


def test_garch_recovery_and_term_structure():
    rng = np.random.default_rng(2)
    T, om, a, b = 15000, 2e-6, 0.08, 0.90
    r, s2 = np.empty(T), om / (1 - a - b)
    for t in range(T):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = om + a * r[t] ** 2 + b * s2
    p = vol.garch_fit(r)
    assert p[1] == pytest.approx(a, abs=0.02) and p[2] == pytest.approx(b, abs=0.03)
    f1, f22 = vol.garch_feature(r, p, 1), vol.garch_feature(r, p, 22)
    longrun = np.log(vol.ANN * p[0] / (1 - p[1] - p[2]))
    assert np.all(np.abs(f22 - longrun) <= np.abs(f1 - longrun) + 1e-12)  # mean reversion with horizon


def test_dm_and_qlike():
    rng = np.random.default_rng(3)
    assert vol.dm(rng.standard_normal(600) - 0.2, 5)[0] < -2
    assert abs(vol.dm(rng.standard_normal(600), 1)[0]) < 3
    assert vol.qlike(np.array([2.0]), np.array([2.0]))[0] == 0
    assert vol.qlike(np.array([2.0]), np.array([1.0]))[0] > vol.qlike(np.array([2.0]), np.array([4.0]))[0] > 0


def test_fast_pipeline(tmp_path):
    from reservoirvol.pipeline import run

    m = run(out=tmp_path, fast=True)
    assert not [c for c in m["checks"] if c["status"] == "FAIL"]
    for f in [
        "report.md",
        "validation.md",
        "metrics.json",
        "forecasts.csv",
        "figures/physics_checks.png",
        "figures/oos_skill.png",
        "figures/forecast_paths.png",
        "figures/sensitivity.png",
    ]:
        assert (tmp_path / f).exists(), f
    assert m["oos"]["mse_ratio"]["h = 1"]["HAR-X (VIX, leverage)"] < 1
