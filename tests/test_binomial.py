import pytest

from pricing import bs_price, crr_price

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_european_converges_to_black_scholes(option, q):
    exact = bs_price(**BASE, q=q, option=option)
    assert crr_price(**BASE, n=2000, q=q, option=option) == pytest.approx(exact, abs=2e-3)


def test_error_shrinks_like_one_over_n():
    exact = bs_price(**BASE)
    err_small = abs(crr_price(**BASE, n=100) - exact)
    err_large = abs(crr_price(**BASE, n=1600) - exact)
    assert err_large < err_small / 8  # 16x more steps, at least 8x smaller error


def test_american_call_without_dividends_equals_european():
    # Early exercise of a call on a non-dividend stock is never optimal.
    european = crr_price(**BASE, n=500)
    american = crr_price(**BASE, n=500, american=True)
    assert american == pytest.approx(european, abs=1e-12)


def test_american_put_benchmark():
    # Widely quoted value for these parameters is 6.0904.
    assert crr_price(**BASE, n=10_000, option="put", american=True) == pytest.approx(6.0904, abs=5e-4)


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_american_put_bounds(K):
    p = dict(BASE, K=K)
    american = crr_price(**p, n=800, option="put", american=True)
    european = crr_price(**p, n=800, option="put")
    assert american >= european
    assert american >= max(K - BASE["S"], 0.0)


def test_american_call_with_dividends_exceeds_european():
    p = dict(BASE, K=90.0)
    assert crr_price(**p, n=800, q=0.08, american=True) > crr_price(**p, n=800, q=0.08) + 0.01
