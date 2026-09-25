"""Volatility data, benchmark models and the out-of-sample engine.

Target: y_t^(h) = log of the average Garman-Klass variance over days
t+1..t+h (h = 1, 5, 22), annualised. The Garman-Klass estimator uses the
daily open, high, low and close and is about seven times more efficient than
the squared close-to-close return.

Every model is a feature map followed by a linear readout that is
re-estimated at the start of each calendar year on all data whose target is
already known (expanding window):

* random walk: the average log variance of the last h days;
* HAR (Corsi 2009): daily, weekly and monthly log variance;
* HAR-X: HAR + log VIX + the last return, split by sign (leverage effect);
* GARCH(1,1) and EWMA: the model's h-day variance forecast, mapped to the
  target by a log-linear calibration (the target excludes overnight moves);
* reservoirs: the state driven by (log variance, return, log VIX), ridge
  readout; an ensemble averages several random reservoirs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

from .reservoir import ridge_fit

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ANN = 252.0


def load(raw: Path = RAW) -> pd.DataFrame:
    px = pd.read_csv(raw / "sp500_ohlc_daily.csv", parse_dates=["date"]).set_index("date")
    vix = pd.read_csv(raw / "vix_daily.csv", parse_dates=["date"]).set_index("date")["vix"]
    o, h, l, c = (np.log(px[k]) for k in ("Open", "High", "Low", "Close"))
    gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    df = pd.DataFrame({"ret": c.diff(), "gk": ANN * gk.clip(lower=1e-8)})
    df["logv"] = np.log(df["gk"])
    df["logvix"] = np.log(vix.reindex(df.index).ffill() / 100) * 2  # log of VIX^2, same units as the target
    return df.dropna()


def targets(df: pd.DataFrame, horizons) -> pd.DataFrame:
    out = {}
    for h in horizons:
        fwd = df["gk"].rolling(h).mean().shift(-h)
        out[h] = np.log(fwd)
    return pd.DataFrame(out)


# Feature maps


def har_features(df: pd.DataFrame, extended: bool = False) -> np.ndarray:
    v = df["logv"]
    cols = [v, v.rolling(5).mean(), v.rolling(22).mean()]
    if extended:
        r = df["ret"] * np.sqrt(ANN)
        cols += [df["logvix"], r.clip(upper=0), r.clip(lower=0)]
    return np.column_stack(cols)


def garch_filter(r, omega, alpha, beta, s0):
    s2 = np.empty(len(r))
    s = s0
    for t in range(len(r)):
        s = omega + alpha * r[t] ** 2 + beta * s  # variance forecast for day t+1 made at t
        s2[t] = s
    return s2


def garch_fit(r: np.ndarray) -> tuple[float, float, float]:
    v = r.var()

    def unpack(z):
        a = 0.3 / (1 + np.exp(-z[0]))
        p = 1 / (1 + np.exp(-z[1]))
        return a, p

    def nll(z):
        a, p = unpack(z)
        if p - a <= 0:
            return 1e12
        s2 = np.concatenate([[v], garch_filter(r, v * (1 - p), a, p - a, v)[:-1]])
        return 0.5 * np.sum(np.log(s2) + r**2 / s2)

    res = minimize(nll, np.array([-0.5, 4.0]), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4})
    a, p = unpack(res.x)
    return v * (1 - p), a, p - a


def garch_feature(r: np.ndarray, params, h: int) -> np.ndarray:
    """log of the average variance forecast over the next h days."""
    omega, a, b = params
    p = a + b
    v = omega / (1 - p)
    s1 = garch_filter(r, omega, a, b, v)
    w = (1 - p**h) / (h * (1 - p))
    return np.log(ANN * (v + (s1 - v) * w))


def ewma_feature(r: np.ndarray, lam: float = 0.94) -> np.ndarray:
    s2 = np.empty(len(r))
    s = r[:250].var()
    for t in range(len(r)):
        s = lam * s + (1 - lam) * r[t] ** 2
        s2[t] = s
    return np.log(ANN * s2)


def reservoir_inputs(df: pd.DataFrame, fit_end: int) -> np.ndarray:
    """Standardised inputs; means and scales from the first `fit_end` rows only."""
    U = np.column_stack([df["logv"], df["ret"], df["logvix"]])
    return (U - U[:fit_end].mean(0)) / U[:fit_end].std(0)


# Out-of-sample engine


def yearly_folds(index: pd.DatetimeIndex, first_year: int):
    for y in sorted({d.year for d in index if d.year >= first_year}):
        te = np.flatnonzero(index.year == y)
        yield y, te


def walk_forward(
    F: np.ndarray, y: np.ndarray, index, h: int, first_year: int, lam: float = 0.0, start: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Predictions for every test day, and the in-sample residual variance of the readout that made them
    (for the log-normal correction when a variance forecast is needed). The readout for year Y is fitted on
    rows t >= start with t + h < first test row: the target must be fully observed before the year starts."""
    pred, svar = np.full(len(y), np.nan), np.full(len(y), np.nan)
    ok = np.isfinite(F).all(1) & np.isfinite(y)
    for _, te in yearly_folds(index, first_year):
        tr = np.arange(start, te[0] - h)
        tr = tr[ok[tr]]
        if lam > 0:
            w, b = ridge_fit(F[tr], y[tr], lam)
        else:
            A = np.column_stack([F[tr], np.ones(len(tr))])
            sol = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            w, b = sol[:-1], sol[-1]
        pred[te] = F[te] @ w + b
        svar[te] = np.var(y[tr] - F[tr] @ w - b)
    return pred, svar


def garch_walk_forward(df: pd.DataFrame, y: np.ndarray, h: int, first_year: int) -> tuple[np.ndarray, np.ndarray]:
    """GARCH parameters and the log-linear calibration are both re-estimated each year."""
    r = df["ret"].to_numpy()
    pred, svar = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for _, te in yearly_folds(df.index, first_year):
        m = r[: te[0]].mean()
        f = garch_feature(r - m, garch_fit(r[: te[0]] - m), h)
        tr = np.arange(30, te[0] - h)
        A = np.column_stack([f[tr], np.ones(len(tr))])
        w = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred[te] = w[0] * f[te] + w[1]
        svar[te] = np.var(y[tr] - A @ w)
    return pred, svar


def dm(d: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano with a Newey-West variance (h - 1 lags) and the Harvey-Leybourne-Newbold correction."""
    d = np.asarray(d, float)
    T = len(d)
    dc = d - d.mean()
    v = dc @ dc / T
    for k in range(1, h):
        v += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / T
    stat = d.mean() / np.sqrt(max(v, 1e-300) / T) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(stat), float(2 * student_t.sf(abs(stat), T - 1))


def qlike(actual_var: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
    x = actual_var / pred_var
    return x - np.log(x) - 1
