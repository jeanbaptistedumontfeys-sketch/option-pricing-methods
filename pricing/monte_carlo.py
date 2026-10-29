"""Monte Carlo pricing with variance reduction.

European options: the terminal price is simulated exactly,
    S_T = S0 exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z),   Z ~ N(0, 1),
and the price is the discounted average payoff.

Variance reduction:
  * Antithetic variates: pair each Z with -Z. The two payoffs are negatively
    correlated, so their average has lower variance than two independent draws.
  * Control variate: the discounted terminal price X = exp(-rT) S_T has the
    known mean S0 exp(-qT). We use Y - beta (X - E[X]) with beta = Cov(Y, X) / Var(X),
    which removes the part of the payoff's variance explained by X.

Arithmetic Asian options have no closed form, but the geometric Asian option
does (the geometric average of lognormals is lognormal). It is very highly
correlated with the arithmetic one, which makes it an excellent control variate.

Note: estimating beta from the same sample adds a bias of order 1 / n_paths,
which is negligible next to the standard error at the sample sizes used here.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .payoffs import check_option, payoff


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self):
        half = 1.96 * self.std_error
        return self.price - half, self.price + half


def _apply_control_variate(y, x, x_mean):
    """Return y - beta (x - E[x]) with the variance-minimising beta."""
    cov = np.cov(y, x)
    beta = cov[0, 1] / cov[1, 1]
    return y - beta * (x - x_mean)


def _summarise(y):
    return MCResult(price=float(y.mean()), std_error=float(y.std(ddof=1) / np.sqrt(len(y))), n_paths=len(y))


def mc_european(
    S0,
    K,
    T,
    r,
    sigma,
    q=0.0,
    option="call",
    n_paths=100_000,
    antithetic=False,
    control_variate=False,
    seed=None,
):
    """Monte Carlo price of a European call or put.

    n_paths is the number of simulated terminal prices, so antithetic runs
    draw n_paths / 2 normals and use each one twice. This keeps comparisons
    fair at equal simulation cost. With antithetic sampling the standard
    error is computed on the pair averages, which are independent.
    """
    check_option(option)
    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    disc = np.exp(-r * T)

    def simulate(z):
        S_T = S0 * np.exp(drift + vol * z)
        return disc * payoff(S_T, K, option), disc * S_T

    if antithetic:
        z = rng.standard_normal(n_paths // 2)
        y_plus, x_plus = simulate(z)
        y_minus, x_minus = simulate(-z)
        y, x = 0.5 * (y_plus + y_minus), 0.5 * (x_plus + x_minus)
    else:
        y, x = simulate(rng.standard_normal(n_paths))

    if control_variate:
        y = _apply_control_variate(y, x, S0 * np.exp(-q * T))

    result = _summarise(y)
    result.n_paths = n_paths
    return result


def geometric_asian_price(S0, K, T, r, sigma, n_dates, q=0.0, option="call"):
    """Closed-form price of a discretely monitored geometric-average Asian option.

    The average is taken over the n_dates equally spaced dates t_i = i T / n_dates.
    log G = (1 / m) sum log S_{t_i} is normal with
        mean     = log S0 + (r - q - sigma^2 / 2) T (m + 1) / (2 m)
        variance = sigma^2 T (m + 1)(2 m + 1) / (6 m^2),
    using sum_{i,j} min(t_i, t_j) = (T / m) m (m + 1)(2 m + 1) / 6.
    """
    check_option(option)
    m = n_dates
    mu = np.log(S0) + (r - q - 0.5 * sigma**2) * T * (m + 1) / (2 * m)
    var = sigma**2 * T * (m + 1) * (2 * m + 1) / (6 * m**2)
    sd = np.sqrt(var)
    d1 = (mu - np.log(K) + var) / sd
    d2 = d1 - sd
    forward_g = np.exp(mu + 0.5 * var)  # E[G]
    disc = np.exp(-r * T)
    if option == "call":
        return float(disc * (forward_g * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(disc * (K * norm.cdf(-d2) - forward_g * norm.cdf(-d1)))


def mc_asian(
    S0,
    K,
    T,
    r,
    sigma,
    n_dates=12,
    q=0.0,
    option="call",
    average="arithmetic",
    n_paths=100_000,
    antithetic=False,
    control_variate=False,
    seed=None,
    batch_size=50_000,
):
    """Monte Carlo price of a discretely monitored Asian option.

    average is 'arithmetic' or 'geometric'. The geometric Asian control
    variate only applies to the arithmetic option. Paths are simulated in
    batches to keep memory use flat.
    """
    check_option(option)
    if average not in ("arithmetic", "geometric"):
        raise ValueError("average must be 'arithmetic' or 'geometric'")
    if control_variate and average != "arithmetic":
        raise ValueError("the geometric control variate is for arithmetic Asians")

    rng = np.random.default_rng(seed)
    dt = T / n_dates
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    disc = np.exp(-r * T)

    def simulate(z):
        log_paths = np.log(S0) + np.cumsum(drift + vol * z, axis=1)
        arith = np.exp(log_paths).mean(axis=1)
        geo = np.exp(log_paths.mean(axis=1))
        target = arith if average == "arithmetic" else geo
        return disc * payoff(target, K, option), disc * payoff(geo, K, option)

    n_draws = n_paths // 2 if antithetic else n_paths
    ys, xs = [], []
    done = 0
    while done < n_draws:
        size = min(batch_size, n_draws - done)
        z = rng.standard_normal((size, n_dates))
        if antithetic:
            y_plus, x_plus = simulate(z)
            y_minus, x_minus = simulate(-z)
            ys.append(0.5 * (y_plus + y_minus))
            xs.append(0.5 * (x_plus + x_minus))
        else:
            y, x = simulate(z)
            ys.append(y)
            xs.append(x)
        done += size

    y, x = np.concatenate(ys), np.concatenate(xs)
    if control_variate:
        y = _apply_control_variate(y, x, geometric_asian_price(S0, K, T, r, sigma, n_dates, q, option))

    result = _summarise(y)
    result.n_paths = n_paths
    return result
