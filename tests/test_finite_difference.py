import warnings

import numpy as np
import pytest

from pricing import bs_greeks, bs_price, crr_price, fd_price

BASE = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
BS_BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_crank_nicolson_matches_black_scholes(option, q):
    res = fd_price(**BASE, q=q, option=option, M=400, N=400)
    assert res.price == pytest.approx(bs_price(**BS_BASE, q=q, option=option), abs=5e-3)


def test_crank_nicolson_is_second_order():
    exact = bs_price(**BS_BASE)
    errors = [abs(fd_price(**BASE, M=m, N=m, S_max=400.0).price - exact) for m in (80, 160, 320)]
    ratios = [errors[0] / errors[1], errors[1] / errors[2]]
    assert all(3.5 < ratio < 4.5 for ratio in ratios)  # halving h divides the error by 4


def test_implicit_scheme_converges():
    exact = bs_price(**BS_BASE)
    assert fd_price(**BASE, theta=1.0, M=400, N=2000).price == pytest.approx(exact, abs=5e-3)


def test_explicit_scheme_stable_grid_works_and_unstable_grid_warns():
    exact = bs_price(**BS_BASE)
    # Stable: dt * sigma^2 * M^2 <= 1 needs N >= 0.04 * 100^2 = 400
    assert fd_price(**BASE, theta=0.0, M=100, N=500, S_max=400.0).price == pytest.approx(exact, abs=0.05)
    with pytest.warns(RuntimeWarning):
        fd_price(**BASE, theta=0.0, M=100, N=50, S_max=400.0)


def test_off_grid_spot_is_interpolated():
    exact = bs_price(103.7, 100.0, 1.0, 0.05, 0.2)
    assert fd_price(103.7, 100.0, 1.0, 0.05, 0.2, M=400, N=400).price == pytest.approx(exact, abs=5e-3)


@pytest.mark.parametrize("option", ["call", "put"])
def test_grid_greeks_match_analytic(option):
    res = fd_price(**BASE, option=option, M=400, N=400)
    greeks = bs_greeks(**BS_BASE, option=option)
    assert res.delta == pytest.approx(greeks["delta"], abs=1e-3)
    assert res.gamma == pytest.approx(greeks["gamma"], abs=1e-4)


def test_rannacher_removes_gamma_oscillations():
    p = dict(BASE, T=0.25)
    S = np.linspace(90, 110, 41)
    exact = bs_greeks(S, 100.0, 0.25, 0.05, 0.2)["gamma"]

    def max_gamma_error(rannacher):
        res = fd_price(**p, M=400, N=10, S_max=400.0, rannacher=rannacher)
        return np.max(np.abs(np.interp(S, res.S, res.gamma_grid()) - exact))

    assert max_gamma_error(rannacher=True) < max_gamma_error(rannacher=False) / 10


def test_american_put_matches_binomial():
    tree = crr_price(**BS_BASE, n=10_000, option="put", american=True)
    res = fd_price(**BASE, option="put", american=True, M=400, N=400)
    assert res.price == pytest.approx(tree, abs=5e-3)
    assert res.price > bs_price(**BS_BASE, option="put")


def test_american_put_exercise_boundary_is_sensible():
    res = fd_price(**BASE, option="put", american=True, M=400, N=400)
    boundary = res.boundary_S
    assert np.all(boundary < BASE["K"])
    # The boundary moves down as time to maturity grows (allow grid-size steps).
    assert np.all(np.diff(boundary) <= res.dS + 1e-12)
    assert boundary[-1] < boundary[0]


def test_american_call_without_dividends_equals_european():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        american = fd_price(**BASE, american=True, M=200, N=200).price
    european = fd_price(**BASE, M=200, N=200).price
    assert american == pytest.approx(european, abs=1e-6)
