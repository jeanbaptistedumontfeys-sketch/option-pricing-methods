"""Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

Over each step dt = T / n the stock moves up by u = exp(sigma sqrt(dt)) or
down by d = 1 / u. The risk-neutral up probability p is chosen so that the
expected growth matches the forward: p = (exp((r - q) dt) - d) / (u - d).
The option is valued by backward induction from the payoff at maturity; for
an American option we take the maximum of the continuation value and the
exercise value at every node.
"""

import numpy as np

from .payoffs import payoff


def crr_price(S, K, T, r, sigma, n, q=0.0, option="call", american=False):
    """Price a vanilla option on an n-step CRR tree."""
    if n < 1:
        raise ValueError("n must be at least 1")
    dt = T / n
    log_u = sigma * np.sqrt(dt)
    u, d = np.exp(log_u), np.exp(-log_u)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("risk-neutral probability is outside (0, 1); increase n")
    disc = np.exp(-r * dt)

    # Node j at step i has had j down moves: S_ij = S u^(i - j) d^j = S exp(log_u (i - 2j)).
    j = np.arange(n + 1)
    values = payoff(S * np.exp(log_u * (n - 2 * j)), K, option)

    for i in range(n - 1, -1, -1):
        # Children of node (i, j) are (i + 1, j) [up] and (i + 1, j + 1) [down].
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            j = np.arange(i + 1)
            np.maximum(values, payoff(S * np.exp(log_u * (i - 2 * j)), K, option), out=values)

    return float(values[0])
