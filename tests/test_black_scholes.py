import numpy as np
import pytest

from pricing import bs_greeks, bs_price, implied_vol

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def test_textbook_values():
    # Textbook example from Hull, Options, Futures and Other Derivatives (S = 42, K = 40)
    assert bs_price(42, 40, 0.5, 0.1, 0.2) == pytest.approx(4.7594, abs=1e-4)
    assert bs_price(42, 40, 0.5, 0.1, 0.2, option="put") == pytest.approx(0.8086, abs=1e-4)
    assert bs_price(**BASE) == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(**BASE, option="put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(q):
    S = np.linspace(60, 140, 9)
    call = bs_price(S, 100, 0.75, 0.04, 0.3, q=q)
    put = bs_price(S, 100, 0.75, 0.04, 0.3, q=q, option="put")
    assert np.allclose(call - put, S * np.exp(-q * 0.75) - 100 * np.exp(-0.04 * 0.75))


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_greeks_match_bump_and_revalue(option, q):
    p = dict(S=105.0, K=100.0, T=0.8, r=0.03, sigma=0.25, q=q, option=option)
    g = bs_greeks(**p)

    def price(**changes):
        return bs_price(**{**p, **changes})

    h = 1e-4
    delta = (price(S=p["S"] + h) - price(S=p["S"] - h)) / (2 * h)
    gamma = (price(S=p["S"] + h) - 2 * price() + price(S=p["S"] - h)) / h**2
    vega = (price(sigma=p["sigma"] + h) - price(sigma=p["sigma"] - h)) / (2 * h)
    rho = (price(r=p["r"] + h) - price(r=p["r"] - h)) / (2 * h)
    theta = -(price(T=p["T"] + h) - price(T=p["T"] - h)) / (2 * h)  # dV/dt = -dV/dT

    assert g["delta"] == pytest.approx(delta, rel=1e-6)
    assert g["gamma"] == pytest.approx(gamma, rel=1e-3)
    assert g["vega"] == pytest.approx(vega, rel=1e-6)
    assert g["rho"] == pytest.approx(rho, rel=1e-6)
    assert g["theta"] == pytest.approx(theta, rel=1e-6)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("K", [60.0, 90.0, 100.0, 120.0, 180.0])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.8, 2.5])
def test_implied_vol_round_trip(option, K, sigma):
    price = bs_price(100.0, K, 1.0, 0.02, sigma, q=0.01, option=option)
    # Price minus its no-arbitrage lower bound. When this is at machine precision
    # the price carries no information about sigma and cannot be inverted.
    forward_gap = 100.0 * np.exp(-0.01) - K * np.exp(-0.02)
    time_value = price - max(forward_gap if option == "call" else -forward_gap, 0.0)
    if time_value < 1e-8:
        pytest.skip("no time value left to invert")
    assert implied_vol(price, 100.0, K, 1.0, 0.02, q=0.01, option=option) == pytest.approx(sigma, rel=1e-7)


def test_implied_vol_rejects_arbitrage_prices():
    with pytest.raises(ValueError):
        implied_vol(0.5, 100, 80, 1.0, 0.05)  # below intrinsic value of the forward
    with pytest.raises(ValueError):
        implied_vol(101.0, 100, 100, 1.0, 0.05)  # call worth more than the stock
