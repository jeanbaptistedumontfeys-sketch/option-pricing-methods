"""Finite-difference solver for the Black-Scholes PDE (European and American).

Writing tau = T - t for the time to maturity, the option value V(S, tau) solves

    dV/dtau = 0.5 sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V,   V(S, 0) = payoff(S).

We put S on a uniform grid S_i = i dS (i = 0..M) on [0, S_max], replace the
S derivatives by central differences, and step forward in tau with the theta
scheme

    (I - theta dt L) V^{n+1} = (I + (1 - theta) dt L) V^n + boundary terms.

theta = 0 is explicit Euler (cheap but only stable for small dt), theta = 1 is
implicit Euler (stable, first order in time) and theta = 0.5 is Crank-Nicolson
(stable, second order in time).

Crank-Nicolson applied to the kinked payoff leaves slowly decaying, high
frequency errors near the strike, which show up as oscillations in delta and
gamma. Rannacher time stepping fixes this by replacing the first two
Crank-Nicolson steps with four implicit Euler half steps.

American options are a linear complementarity problem at every time step:
find V with A V >= b, V >= payoff, and equality in at least one of the two.
We solve it with projected successive over-relaxation (PSOR).
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded

from .payoffs import check_option, payoff


@dataclass
class FDResult:
    """Output of fd_price: the price at S0 plus the whole grid for Greeks and plots."""

    price: float
    S0: float
    S: np.ndarray
    V: np.ndarray
    # Early exercise boundary (American only): S*(tau) at each time step.
    boundary_tau: np.ndarray = field(default_factory=lambda: np.array([]))
    boundary_S: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def dS(self):
        return self.S[1] - self.S[0]

    def delta_grid(self):
        """Central-difference delta at every interior node (NaN at the two edges)."""
        out = np.full_like(self.V, np.nan)
        out[1:-1] = (self.V[2:] - self.V[:-2]) / (2.0 * self.dS)
        return out

    def gamma_grid(self):
        """Central-difference gamma at every interior node (NaN at the two edges)."""
        out = np.full_like(self.V, np.nan)
        out[1:-1] = (self.V[2:] - 2.0 * self.V[1:-1] + self.V[:-2]) / self.dS**2
        return out

    @property
    def delta(self):
        return float(np.interp(self.S0, self.S[1:-1], self.delta_grid()[1:-1]))

    @property
    def gamma(self):
        return float(np.interp(self.S0, self.S[1:-1], self.gamma_grid()[1:-1]))


def _boundary_values(tau, K, r, q, S_max, option, american):
    """Values at S = 0 and S = S_max for time to maturity tau."""
    if option == "call":
        low = 0.0
        high = S_max * np.exp(-q * tau) - K * np.exp(-r * tau)
        if american:
            high = max(high, S_max - K)
    else:
        low = K if american else K * np.exp(-r * tau)
        high = 0.0
    return low, high


def _solve_tridiagonal(lower, diag, upper, rhs):
    """Solve a tridiagonal system. lower[k] multiplies x[k-1], upper[k] multiplies x[k+1]."""
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


def _psor(lower, diag, upper, rhs, x0, g, omega, tol, max_iter):
    """Projected SOR for  A x >= rhs, x >= g, (A x - rhs)(x - g) = 0  with A tridiagonal.

    Each sweep is a Gauss-Seidel update over-relaxed by omega and then
    projected onto x >= g. Python lists are used because the loop is scalar.
    """
    lo, di, up = lower.tolist(), diag.tolist(), upper.tolist()
    b, gl = rhs.tolist(), g.tolist()
    x = np.maximum(x0, g).tolist()
    n = len(x)
    for _ in range(max_iter):
        err = 0.0
        for k in range(n):
            s = b[k]
            if k > 0:
                s -= lo[k] * x[k - 1]
            if k < n - 1:
                s -= up[k] * x[k + 1]
            new = x[k] + omega * (s / di[k] - x[k])
            if new < gl[k]:
                new = gl[k]
            diff = new - x[k]
            err += diff * diff
            x[k] = new
        if err < tol * tol:
            return np.array(x)
    raise RuntimeError("PSOR did not converge; try a different omega or more iterations")


def fd_price(
    S0,
    K,
    T,
    r,
    sigma,
    q=0.0,
    option="call",
    american=False,
    M=200,
    N=200,
    theta=0.5,
    rannacher=True,
    S_max=None,
    omega=1.2,
    tol=1e-9,
    max_iter=10_000,
):
    """Price a vanilla option by finite differences.

    M is the number of space intervals and N the number of time steps.
    Returns an FDResult with the price and the full grid at tau = T.
    """
    check_option(option)
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be in [0, 1]")
    if S_max is None:
        S_max = max(S0, K) * max(4.0, float(np.exp(5.0 * sigma * np.sqrt(T))))
    if not 0.0 < S0 < S_max:
        raise ValueError("S0 must lie strictly inside (0, S_max)")

    S = np.linspace(0.0, S_max, M + 1)
    i = np.arange(1, M, dtype=float)
    # Row i of L:  a_i V_{i-1} + b_i V_i + c_i V_{i+1}   (using S_i = i dS)
    a = 0.5 * sigma**2 * i**2 - 0.5 * (r - q) * i
    b = -(sigma**2) * i**2 - r
    c = 0.5 * sigma**2 * i**2 + 0.5 * (r - q) * i

    g = payoff(S, K, option)
    V = g.copy()
    dt = T / N

    if theta == 0.5 and rannacher and N >= 2:
        steps = [(0.5 * dt, 1.0)] * 4 + [(dt, 0.5)] * (N - 2)
    else:
        steps = [(dt, theta)] * N

    if theta == 0.0 and dt * (sigma**2 * (M - 1) ** 2 + r) > 1.0:
        warnings.warn(
            "explicit scheme is likely unstable for this grid: reduce dt or use theta > 0",
            RuntimeWarning,
            stacklevel=2,
        )

    tau = 0.0
    boundary_tau, boundary_S = [], []
    for step_dt, th in steps:
        tau += step_dt
        low, high = _boundary_values(tau, K, r, q, S_max, option, american)

        # Explicit part, using the full old vector (boundaries included).
        rhs = V[1:-1] + (1.0 - th) * step_dt * (a * V[:-2] + b * V[1:-1] + c * V[2:])
        lower = -th * step_dt * a
        diag = 1.0 - th * step_dt * b
        upper = -th * step_dt * c
        # Move the known new boundary values to the right-hand side.
        rhs[0] -= lower[0] * low
        rhs[-1] -= upper[-1] * high

        if th == 0.0:
            interior = np.maximum(rhs, g[1:-1]) if american else rhs
        elif american:
            interior = _psor(lower, diag, upper, rhs, V[1:-1], g[1:-1], omega, tol, max_iter)
        else:
            interior = _solve_tridiagonal(lower, diag, upper, rhs)

        V = np.concatenate(([low], interior, [high]))

        if american:
            exercised = (V[1:-1] <= g[1:-1] + 1e-12) & (g[1:-1] > 0.0)
            S_int = S[1:-1]
            if exercised.any():
                s_star = S_int[exercised].max() if option == "put" else S_int[exercised].min()
            else:
                s_star = np.nan
            boundary_tau.append(tau)
            boundary_S.append(s_star)

    price = float(CubicSpline(S, V)(S0))
    return FDResult(
        price=price,
        S0=S0,
        S=S,
        V=V,
        boundary_tau=np.array(boundary_tau),
        boundary_S=np.array(boundary_S),
    )
