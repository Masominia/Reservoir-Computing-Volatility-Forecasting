"""Reservoir computing: a fixed random dynamical system driven by the input,
with only a linear readout trained (ridge regression).

Two reservoirs with the same interface:

* **EchoStateNetwork** (Jaeger 2001): leaky tanh units, sparse random
  recurrent matrix rescaled to a chosen spectral radius,

      x_t = (1 - a) x_{t-1} + a tanh(W x_{t-1} + W_in u_t + b);

* **RingReservoir**: the discrete model of an opto-electronic delay-line
  reservoir (Appeltant et al. 2011; Paquot et al. 2012). A single nonlinear
  node with delayed feedback is time-multiplexed into N virtual nodes, which
  makes the coupling a ring; the nonlinearity is the sine response of a
  Mach-Zehnder modulator and the random input mask plays the role of W_in,

      x_{i,t} = sin(alpha x_{i-1,t-1} + beta (M u_t)_i + phi).

  This is hardware that exists: a laser, a modulator, a fibre spool and a
  photodiode.

Also here: ridge readout, linear memory capacity (Jaeger 2002) and the
Mackey-Glass chaotic benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EchoStateNetwork:
    n: int = 200
    spectral_radius: float = 0.9
    leak: float = 0.3
    input_scale: float = 0.5
    density: float = 0.1
    seed: int = 0

    def __post_init__(self):
        self._built_for = None

    def _build(self, n_in: int):
        rng = np.random.default_rng(self.seed)
        W = rng.standard_normal((self.n, self.n)) * (rng.random((self.n, self.n)) < self.density)
        W *= self.spectral_radius / np.abs(np.linalg.eigvals(W)).max()
        self.W = W
        self.W_in = self.input_scale * rng.uniform(-1, 1, (self.n, n_in))
        self.b = 0.1 * rng.uniform(-1, 1, self.n)
        self._built_for = n_in

    def run(self, U: np.ndarray) -> np.ndarray:
        """States (T, n) for inputs (T, n_in), zero initial state."""
        U = np.atleast_2d(U.T).T
        if self._built_for != U.shape[1]:
            self._build(U.shape[1])
        X = np.empty((len(U), self.n))
        x = np.zeros(self.n)
        drive = U @ self.W_in.T + self.b
        a = self.leak
        for t in range(len(U)):
            x = (1 - a) * x + a * np.tanh(self.W @ x + drive[t])
            X[t] = x
        return X


@dataclass
class RingReservoir:
    n: int = 200
    feedback: float = 0.8  # alpha
    input_scale: float = 0.5  # beta
    phase: float = 0.2  # phi
    seed: int = 0

    def run(self, U: np.ndarray) -> np.ndarray:
        U = np.atleast_2d(U.T).T
        rng = np.random.default_rng(self.seed)
        mask = rng.choice([-1.0, 1.0], (self.n, U.shape[1]))
        drive = self.input_scale * (U @ mask.T) + self.phase
        X = np.empty((len(U), self.n))
        x = np.zeros(self.n)
        for t in range(len(U)):
            x = np.sin(self.feedback * np.roll(x, 1) + drive[t])
            X[t] = x
        return X


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Ridge regression with an unpenalised intercept; lam is relative to the mean feature variance."""
    mx, my = X.mean(0), y.mean(0)
    Xc = X - mx
    G = Xc.T @ Xc
    G[np.diag_indices_from(G)] += lam * np.trace(G) / len(G)
    w = np.linalg.solve(G, Xc.T @ (y - my))
    return w, my - mx @ w


def memory_capacity(res, k_max: int, T: int = 4000, washout: int = 200, lam: float = 1e-6, seed: int = 0):
    """MC_k = squared correlation between u_{t-k} and its best linear reconstruction from the state,
    for iid uniform input (train on the first half, measure on the second). Sum over k <= number of nodes."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, T)
    X = res.run(u[:, None])
    half = (T + washout) // 2
    mc = np.empty(k_max)
    for k in range(1, k_max + 1):
        tr = slice(max(washout, k), half)
        te = slice(half, T)
        w, b = ridge_fit(X[tr], u[tr.start - k : tr.stop - k], lam)
        mc[k - 1] = np.corrcoef(X[te] @ w + b, u[te.start - k : te.stop - k])[0, 1] ** 2
    return mc


def mackey_glass(T: int, tau: int = 17, dt: float = 0.1, sample: int = 10, seed: int = 0) -> np.ndarray:
    """Mackey-Glass delay equation dx/dt = 0.2 x(t - tau) / (1 + x(t - tau)^10) - 0.1 x(t), RK4, chaotic for
    tau = 17. Returns T samples taken every `sample` integration steps after a transient."""
    rng = np.random.default_rng(seed)
    d = int(tau / dt)
    n = (T + 500) * sample + d
    x = np.empty(n)
    x[: d + 1] = 1.2 + 0.05 * rng.standard_normal(d + 1)

    def f(xt, xd):
        return 0.2 * xd / (1 + xd**10) - 0.1 * xt

    for t in range(d, n - 1):
        xd, xd1 = x[t - d], x[t - d + 1]
        xm = 0.5 * (xd + xd1)
        k1 = f(x[t], xd)
        k2 = f(x[t] + 0.5 * dt * k1, xm)
        k3 = f(x[t] + 0.5 * dt * k2, xm)
        k4 = f(x[t] + dt * k3, xd1)
        x[t + 1] = x[t] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x[d + 500 * sample :: sample][:T]


def direct_forecast_nrmse(
    res, series: np.ndarray, horizon: int, train: int, washout: int = 200, lam: float = 1e-8
) -> float:
    """Train a readout to predict series[t + horizon] from the state at t; NRMSE on the remainder."""
    s = (series - series[:train].mean()) / series[:train].std()
    X = res.run(s[:, None])
    y = s[horizon:]
    X = X[: len(y)]
    w, b = ridge_fit(X[washout:train], y[washout:train], lam)
    err = X[train:] @ w + b - y[train:]
    return float(np.sqrt(np.mean(err**2)) / y[train:].std())


def ar_forecast_nrmse(series: np.ndarray, horizon: int, train: int, lags: int = 50) -> float:
    """Linear benchmark: direct regression of series[t + horizon] on the last `lags` values."""
    s = (series - series[:train].mean()) / series[:train].std()
    X = np.column_stack([s[lags - 1 - j : len(s) - horizon - j] for j in range(lags)])
    y = s[lags - 1 + horizon :]
    k = train - lags
    w, b = ridge_fit(X[:k], y[:k], 1e-8)
    err = X[k:] @ w + b - y[k:]
    return float(np.sqrt(np.mean(err**2)) / y[k:].std())
