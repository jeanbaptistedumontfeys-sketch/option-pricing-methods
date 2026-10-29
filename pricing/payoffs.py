"""Vanilla payoffs shared by all pricing methods."""

import numpy as np


def check_option(option):
    """Raise if option is not 'call' or 'put'."""
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")


def payoff(S, K, option):
    """Payoff of a vanilla call or put at spot S (scalar or array)."""
    check_option(option)
    if option == "call":
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)
