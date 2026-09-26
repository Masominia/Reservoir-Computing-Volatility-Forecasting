import numpy as np
import pytest

from reservoirvol import reservoir as rc


def test_ridge_matches_ols_and_shrinks():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + 0.7 + 0.01 * rng.standard_normal(400)
    w, b = rc.ridge_fit(X, y, 1e-12)
    assert np.allclose(w, [1, -2, 0.5, 0, 3], atol=0.01) and b == pytest.approx(0.7, abs=0.01)
    assert np.linalg.norm(rc.ridge_fit(X, y, 10.0)[0]) < np.linalg.norm(w)


def test_echo_state_property():
    """Two different initial conditions converge under the same input (fading memory) when the radius is < 1."""
    esn = rc.EchoStateNetwork(80, 0.8, 0.5, 0.5, seed=1)
    u = np.random.default_rng(1).standard_normal((400, 1))
    a = esn.run(u)
    b = esn.run(np.vstack([5 * np.ones((1, 1)), u[1:]]))  # perturbed first input
    assert np.abs(a[1] - b[1]).max() > 1e-3 and np.abs(a[-1] - b[-1]).max() < 1e-8
    assert np.abs(np.linalg.eigvals(esn.W)).max() == pytest.approx(0.8)
    assert np.abs(a).max() <= 1.0


def test_reservoirs_are_reproducible_and_seed_dependent():
    u = np.random.default_rng(2).standard_normal((50, 3))
    for cls in (rc.EchoStateNetwork, rc.RingReservoir):
        assert np.array_equal(cls(30, seed=3).run(u), cls(30, seed=3).run(u))
        assert not np.allclose(cls(30, seed=3).run(u), cls(30, seed=4).run(u))


def test_ring_topology():
    """With zero input and phase, each virtual node only inherits from its neighbour: sin(alpha x_{i-1})."""
    ring = rc.RingReservoir(5, feedback=0.9, input_scale=0.0, phase=0.3, seed=0)
    X = ring.run(np.zeros((3, 1)))
    assert np.allclose(X[0], np.sin(0.3))
    assert np.allclose(X[1], np.sin(0.9 * np.roll(X[0], 1) + 0.3))


def test_memory_capacity_bounds_and_edge_of_chaos():
    n = 40
    mc = rc.memory_capacity(rc.RingReservoir(n, 0.9, 0.1, 0.0, seed=0), 80, T=3000)
    assert mc[0] > 0.99 and mc[-1] < 0.05 and mc.sum() <= n
    low = rc.memory_capacity(rc.RingReservoir(n, 0.3, 0.1, 0.0, seed=0), 80, T=3000).sum()
    chaotic = rc.memory_capacity(rc.RingReservoir(n, 1.4, 0.1, 0.0, seed=0), 80, T=3000).sum()
    assert low < mc.sum() and chaotic < 0.3 * mc.sum()


def test_mackey_glass_is_bounded_chaos_and_predictable_by_reservoir():
    mg = rc.mackey_glass(3000)
    assert 0.2 < mg.min() < mg.max() < 1.5
    other = rc.mackey_glass(3000, seed=1)
    assert np.abs(mg - other).max() > 0.1  # sensitive dependence on the initial history
    esn = rc.EchoStateNetwork(200, 0.9, 0.5, 0.5, seed=0)
    assert rc.direct_forecast_nrmse(esn, mg, 20, 1500) < 0.1 * rc.ar_forecast_nrmse(mg, 20, 1500)
