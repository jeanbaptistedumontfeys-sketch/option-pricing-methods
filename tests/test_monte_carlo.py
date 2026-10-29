import pytest

from pricing import bs_price, geometric_asian_price, mc_asian, mc_european

BASE = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
BS_BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
@pytest.mark.parametrize("control_variate", [False, True])
def test_european_within_four_standard_errors(option, antithetic, control_variate):
    res = mc_european(
        **BASE, option=option, n_paths=200_000, antithetic=antithetic,
        control_variate=control_variate, seed=7,
    )
    exact = bs_price(**BS_BASE, option=option)
    assert abs(res.price - exact) < 4 * res.std_error


def test_variance_reduction_lowers_standard_error():
    se = {
        flags: mc_european(**BASE, n_paths=200_000, antithetic=flags[0], control_variate=flags[1], seed=3).std_error
        for flags in [(False, False), (True, False), (False, True), (True, True)]
    }
    plain = se[(False, False)]
    assert se[(True, False)] < plain
    assert se[(False, True)] < plain / 2
    assert se[(True, True)] < plain / 2


def test_geometric_asian_with_one_date_is_black_scholes():
    assert geometric_asian_price(**BASE, n_dates=1) == pytest.approx(bs_price(**BS_BASE), abs=1e-10)
    put = geometric_asian_price(**BASE, n_dates=1, option="put")
    assert put == pytest.approx(bs_price(**BS_BASE, option="put"), abs=1e-10)


@pytest.mark.parametrize("option", ["call", "put"])
def test_geometric_asian_closed_form_matches_simulation(option):
    res = mc_asian(**BASE, n_dates=12, option=option, average="geometric", n_paths=200_000, seed=11)
    exact = geometric_asian_price(**BASE, n_dates=12, option=option)
    assert abs(res.price - exact) < 4 * res.std_error


def test_arithmetic_asian_control_variate():
    plain = mc_asian(**BASE, n_dates=12, n_paths=200_000, seed=5)
    cv = mc_asian(**BASE, n_dates=12, n_paths=200_000, control_variate=True, seed=5)
    # Same answer, far smaller error.
    assert abs(plain.price - cv.price) < 4 * plain.std_error
    assert cv.std_error < plain.std_error / 20
    # Arithmetic mean >= geometric mean, so the arithmetic call is worth more.
    assert cv.price > geometric_asian_price(**BASE, n_dates=12)


def test_ci95_contains_price():
    res = mc_european(**BASE, n_paths=10_000, seed=1)
    lo, hi = res.ci95
    assert lo < res.price < hi
