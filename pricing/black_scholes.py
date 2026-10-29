"""Closed-form Black-Scholes prices, Greeks and implied volatility.

Model: under the risk-neutral measure the stock follows geometric Brownian
motion, dS = (r - q) S dt + sigma S dW, with a constant risk-free rate r, a
continuous dividend yield q and a constant volatility sigma.

bs_price and bs_greeks accept scalars or NumPy arrays (they broadcast).
implied_vol works on one price at a time.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .payoffs import check_option


def d1_d2(S, K, T, r, sigma, q=0.0):
    """Return the d1 and d2 terms of the Black-Scholes formula."""
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(S, K, T, r, sigma, q=0.0, option="call"):
    """Black-Scholes price of a European call or put."""
    check_option(option)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    pv_s = S * np.exp(-q * T)  # spot net of dividends paid before T
    pv_k = K * np.exp(-r * T)  # present value of the strike
    if option == "call":
        return pv_s * norm.cdf(d1) - pv_k * norm.cdf(d2)
    return pv_k * norm.cdf(-d2) - pv_s * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, q=0.0, option="call"):
    """Analytic Greeks of a European call or put.

    Returns a dict with delta, gamma, vega, theta and rho. Vega and rho are
    per unit change (multiply by 0.01 for "per 1%"), theta is the change in
    value per year of calendar time (dV/dt, usually negative).
    """
    check_option(option)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    sqrt_t = np.sqrt(T)
    df_q = np.exp(-q * T)
    df_r = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = df_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * df_q * pdf_d1 * sqrt_t
    time_decay = -S * df_q * pdf_d1 * sigma / (2.0 * sqrt_t)

    if option == "call":
        delta = df_q * norm.cdf(d1)
        theta = time_decay - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
        rho = K * T * df_r * norm.cdf(d2)
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = time_decay + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)
        rho = -K * T * df_r * norm.cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price, S, K, T, r, q=0.0, option="call", tol=1e-12):
    """Volatility that makes the Black-Scholes price equal to the given price.

    The Black-Scholes price is strictly increasing in sigma, so the root is
    unique. We bracket it and solve with Brent's method.
    """
    check_option(option)
    pv_s = S * np.exp(-q * T)
    pv_k = K * np.exp(-r * T)
    if option == "call":
        lower, upper = max(pv_s - pv_k, 0.0), pv_s
    else:
        lower, upper = max(pv_k - pv_s, 0.0), pv_k
    if not lower < price < upper:
        raise ValueError(
            f"price {price:.6g} is outside the no-arbitrage bounds ({lower:.6g}, {upper:.6g})"
        )

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, q, option) - price

    lo, hi = 1e-8, 1.0
    while objective(hi) < 0.0:
        hi *= 2.0
        if hi > 100.0:
            raise ValueError("could not bracket the implied volatility")
    return float(brentq(objective, lo, hi, xtol=tol, maxiter=500))
