"""Reproduce every table and figure in the README.

Run from the repository root:
    python experiments/run_experiments.py

Writes PNG figures to figures/ and Markdown tables to results/results.md.
Runtimes depend on the machine; the prices and errors do not (fixed seeds).
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pricing import (  # noqa: E402
    bs_greeks,
    bs_price,
    crr_price,
    fd_price,
    geometric_asian_price,
    mc_asian,
    mc_european,
)

FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2
SEED = 42

COLORS = {
    "tree": "#4C72B0",
    "cn_r": "#DD8452",
    "cn": "#C44E52",
    "implicit": "#55A868",
    "exact": "#333333",
    "plain": "#8172B3",
    "anti": "#4C72B0",
    "cv": "#DD8452",
    "both": "#55A868",
}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
    }
)

lines = []


def emit(text=""):
    print(text)
    lines.append(text)


def timed(fn, *args, repeat=3, **kwargs):
    """Return (result, best wall time in ms) over a few runs."""
    best, out = np.inf, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return out, 1000.0 * best


def slope(x, y):
    """Least-squares slope in log-log space = empirical order of convergence."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


# 1. Headline comparison -------------------------------------------------------
def benchmark():
    emit("## 1. One option, four methods")
    emit()
    emit(f"S0 = {S0:g}, K = {K:g}, T = {T:g}, r = {R:g}, sigma = {SIGMA:g}, no dividends.")
    emit()

    ref_call = bs_price(S0, K, T, R, SIGMA)
    ref_put = bs_price(S0, K, T, R, SIGMA, option="put")
    # American reference: average of two very fine trees (cancels the odd/even wobble).
    ref_am = 0.5 * (
        crr_price(S0, K, T, R, SIGMA, 20_000, option="put", american=True)
        + crr_price(S0, K, T, R, SIGMA, 20_001, option="put", american=True)
    )

    emit("| Method | European call | European put | American put | Time (call) |")
    emit("|---|---|---|---|---|")
    emit(f"| Black-Scholes closed form | {ref_call:.4f} | {ref_put:.4f} | n/a | "
         f"{timed(bs_price, S0, K, T, R, SIGMA, repeat=20)[1]:.3f} ms |")

    c, t_c = timed(crr_price, S0, K, T, R, SIGMA, 1000)
    p = crr_price(S0, K, T, R, SIGMA, 1000, option="put")
    a = crr_price(S0, K, T, R, SIGMA, 1000, option="put", american=True)
    emit(f"| CRR tree, 1,000 steps | {c:.4f} | {p:.4f} | {a:.4f} | {t_c:.1f} ms |")

    c, t_c = timed(lambda: fd_price(S0, K, T, R, SIGMA, M=400, N=400).price)
    p = fd_price(S0, K, T, R, SIGMA, option="put", M=400, N=400).price
    a = fd_price(S0, K, T, R, SIGMA, option="put", american=True, M=400, N=400).price
    emit(f"| Crank-Nicolson + Rannacher, 400 x 400 (PSOR for American) | {c:.4f} | {p:.4f} | {a:.4f} | {t_c:.1f} ms |")

    mc_c, t_c = timed(mc_european, S0, K, T, R, SIGMA, n_paths=1_000_000,
                      antithetic=True, control_variate=True, seed=SEED)
    mc_p = mc_european(S0, K, T, R, SIGMA, option="put", n_paths=1_000_000,
                       antithetic=True, control_variate=True, seed=SEED)
    emit(f"| Monte Carlo, 1M paths, antithetic + control variate | {mc_c.price:.4f} (SE {mc_c.std_error:.4f}) | "
         f"{mc_p.price:.4f} (SE {mc_p.std_error:.4f}) | n/a | {t_c:.1f} ms |")
    emit(f"| Reference | {ref_call:.4f} | {ref_put:.4f} | {ref_am:.4f} (tree, 20,000 steps) | |")
    emit()
    return ref_am


# 2. Convergence of the grid methods and of Monte Carlo -------------------------
def convergence():
    emit("## 2. Convergence")
    emit()
    exact = bs_price(S0, K, T, R, SIGMA)

    n_tree = np.unique(np.round(np.logspace(1, np.log10(2000), 70)).astype(int))
    err_tree = np.array([abs(crr_price(S0, K, T, R, SIGMA, n) - exact) for n in n_tree])

    sizes = np.array([20, 40, 80, 160, 320, 640, 1280])  # multiples of 4 so S0 = K = 100 is a node
    schemes = {
        "cn_r": dict(theta=0.5, rannacher=True),
        "cn": dict(theta=0.5, rannacher=False),
        "implicit": dict(theta=1.0),
    }
    err_fd = {
        name: np.array([abs(fd_price(S0, K, T, R, SIGMA, M=m, N=m, S_max=400.0, **kw).price - exact)
                        for m in sizes])
        for name, kw in schemes.items()
    }

    paths = np.array([1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000])
    variants = {"plain": (False, False), "anti": (True, False), "cv": (False, True), "both": (True, True)}
    se = {name: np.array([mc_european(S0, K, T, R, SIGMA, n_paths=n, antithetic=a, control_variate=c,
                                      seed=SEED).std_error for n in paths])
          for name, (a, c) in variants.items()}

    emit("Empirical order = slope of log(error) against log(steps). Grid sizes use M = N.")
    emit()
    emit("| Method | Error at smallest size | Error at largest size | Empirical order |")
    emit("|---|---|---|---|")
    emit(f"| CRR tree (10 to 2,000 steps) | {err_tree[0]:.2e} | {err_tree[-1]:.2e} | {slope(n_tree, err_tree):.2f} |")
    labels = {"cn_r": "Crank-Nicolson + Rannacher", "cn": "Crank-Nicolson, no smoothing", "implicit": "Implicit Euler"}
    for name in schemes:
        emit(f"| {labels[name]} (M = N = {sizes[0]} to {sizes[-1]}) | {err_fd[name][0]:.2e} | {err_fd[name][-1]:.2e} | "
             f"{slope(sizes, err_fd[name]):.2f} |")
    emit(f"| Monte Carlo standard error, plain (1k to 1M paths) | {se['plain'][0]:.2e} | {se['plain'][-1]:.2e} | "
         f"{slope(paths, se['plain']):.2f} |")
    emit()
    emit(f"Error ratio between the two finest grids ({sizes[-2]} and {sizes[-1]}): "
         + ", ".join(f"{labels[name]} {err_fd[name][-2] / err_fd[name][-1]:.2f}" for name in schemes)
         + " (4 = second order, 2 = first order).")
    emit()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.loglog(n_tree, err_tree, color=COLORS["tree"], lw=1, label="CRR tree")
    ax1.loglog(sizes, err_fd["implicit"], "o-", color=COLORS["implicit"], label="Implicit Euler")
    ax1.loglog(sizes, err_fd["cn"], "s--", color=COLORS["cn"], mfc="none", ms=9, label="Crank-Nicolson")
    ax1.loglog(sizes, err_fd["cn_r"], "o-", color=COLORS["cn_r"], label="Crank-Nicolson + Rannacher")
    # Reference slopes, anchored so they pass just below the data.
    guide = sizes.astype(float)
    c1 = 0.5 * err_tree[-1] * n_tree[-1]
    c2 = 0.5 * err_fd["cn_r"][-1] * sizes[-1] ** 2
    ax1.loglog(guide, c1 / guide, ":", color="grey", lw=1)
    ax1.loglog(guide, c2 / guide**2, ":", color="grey", lw=1)
    ax1.text(guide[0], c1 / guide[0], "slope -1  ", color="grey", va="center", ha="right")
    ax1.text(guide[0], c2 / guide[0] ** 2, "slope -2  ", color="grey", va="center", ha="right")
    ax1.set_xlabel("tree steps, or grid size M = N")
    ax1.set_ylabel("absolute error vs Black-Scholes")
    ax1.set_title("Tree and PDE: European call")
    ax1.legend(fontsize=9)

    names = {"plain": "Plain", "anti": "Antithetic", "cv": "Control variate", "both": "Antithetic + control variate"}
    for name in variants:
        ax2.loglog(paths, se[name], "o-", color=COLORS[name], label=names[name])
    ax2.set_xlabel("number of simulated prices")
    ax2.set_ylabel("standard error")
    ax2.set_title("Monte Carlo: European call (all slopes -1/2)")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "convergence.png")
    plt.close(fig)


# 3. Variance reduction ------------------------------------------------------------
def variance_reduction():
    emit("## 3. Variance reduction at 100,000 paths")
    emit()
    emit("Variance reduction factor = (plain SE / method SE)². Efficiency also divides by the extra runtime.")
    emit()
    n = 100_000
    exact_call = bs_price(S0, K, T, R, SIGMA)

    emit(f"**European call** (exact {exact_call:.4f})")
    emit()
    emit("| Method | Price | Std error | Variance reduction | Efficiency gain |")
    emit("|---|---|---|---|---|")
    rows = [("Plain", False, False), ("Antithetic", True, False), ("Control variate (S_T)", False, True),
            ("Antithetic + control variate", True, True)]
    base_se = base_t = None
    for label, a, c in rows:
        res, t = timed(mc_european, S0, K, T, R, SIGMA, n_paths=n, antithetic=a, control_variate=c, seed=SEED, repeat=5)
        if base_se is None:
            base_se, base_t = res.std_error, t
        vr = (base_se / res.std_error) ** 2
        emit(f"| {label} | {res.price:.4f} | {res.std_error:.4f} | {vr:.1f}x | {vr * base_t / t:.1f}x |")
    emit()

    geo = geometric_asian_price(S0, K, T, R, SIGMA, n_dates=12)
    emit(f"**Arithmetic Asian call, 12 monthly averaging dates** (no closed form; geometric Asian closed form = {geo:.4f})")
    emit()
    emit("| Method | Price | Std error | Variance reduction | Efficiency gain |")
    emit("|---|---|---|---|---|")
    rows = [("Plain", False, False), ("Antithetic", True, False), ("Geometric Asian control variate", False, True),
            ("Antithetic + control variate", True, True)]
    base_se = base_t = None
    for label, a, c in rows:
        res, t = timed(mc_asian, S0, K, T, R, SIGMA, n_dates=12, n_paths=n, antithetic=a, control_variate=c,
                       seed=SEED, repeat=3)
        if base_se is None:
            base_se, base_t = res.std_error, t
        vr = (base_se / res.std_error) ** 2
        emit(f"| {label} | {res.price:.4f} | {res.std_error:.5f} | {vr:.0f}x | {vr * base_t / t:.0f}x |")
    emit()


# 4. American put ------------------------------------------------------------------
def american():
    emit("## 4. American put: early exercise premium")
    emit()
    emit("| Strike | European (closed form) | American (CRR, 10,000 steps) | American (FD + PSOR, 400 x 400) | Premium |")
    emit("|---|---|---|---|---|")
    for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
        eu = bs_price(S0, strike, T, R, SIGMA, option="put")
        tree = crr_price(S0, strike, T, R, SIGMA, 10_000, option="put", american=True)
        fd = fd_price(S0, strike, T, R, SIGMA, option="put", american=True, M=400, N=400).price
        emit(f"| {strike:g} | {eu:.4f} | {tree:.4f} | {fd:.4f} | {tree - eu:.4f} |")
    emit()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for sigma, ls in ((0.2, "-"), (0.4, "--")):
        res = fd_price(S0, K, T, R, sigma, option="put", american=True, M=800, N=800, S_max=400.0)
        t = T - res.boundary_tau
        ax.plot(t, res.boundary_S, ls, color=COLORS["tree"], label=rf"$\sigma$ = {sigma:g}")
        if sigma == 0.2:
            ax.fill_between(t, 0, res.boundary_S, color=COLORS["tree"], alpha=0.08)
    ax.axhline(K, color="grey", lw=0.8, ls=":")
    ax.text(0.02, K + 0.6, "strike", color="grey")
    ax.text(0.03, 76, r"shaded: exercise region for $\sigma$ = 0.2", color=COLORS["tree"])
    ax.text(0.35, 93, "hold (continuation region)", color="#555555")
    ax.set_ylim(55, 103)
    ax.set_xlim(0, T)
    ax.set_xlabel("calendar time t (years); maturity at t = 1")
    ax.set_ylabel("critical stock price S*(t)")
    ax.set_title("American put early exercise boundary (K = 100, r = 5%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "exercise_boundary.png")
    plt.close(fig)


# 5. Greeks from the grid ----------------------------------------------------------
def greeks():
    emit("## 5. Greeks from the PDE grid")
    emit()
    g = bs_greeks(S0, K, T, R, SIGMA)
    res = fd_price(S0, K, T, R, SIGMA, M=400, N=400)
    emit("European call, 400 x 400 grid, central differences on the final grid:")
    emit()
    emit("| Greek | Analytic | Finite-difference grid | Abs error |")
    emit("|---|---|---|---|")
    emit(f"| Delta | {g['delta']:.5f} | {res.delta:.5f} | {abs(g['delta'] - res.delta):.1e} |")
    emit(f"| Gamma | {g['gamma']:.5f} | {res.gamma:.5f} | {abs(g['gamma'] - res.gamma):.1e} |")
    emit()

    # Short maturity and large time steps make the Crank-Nicolson oscillations visible.
    t_short, n_steps = 0.25, 10
    S = np.linspace(80, 120, 401)
    exact = bs_greeks(S, K, t_short, R, SIGMA)
    plain = fd_price(S0, K, t_short, R, SIGMA, M=400, N=n_steps, S_max=400.0, rannacher=False)
    smooth = fd_price(S0, K, t_short, R, SIGMA, M=400, N=n_steps, S_max=400.0, rannacher=True)
    err_plain = np.max(np.abs(np.interp(S, plain.S, plain.gamma_grid()) - exact["gamma"]))
    err_smooth = np.max(np.abs(np.interp(S, smooth.S, smooth.gamma_grid()) - exact["gamma"]))
    emit(f"Short-dated call (T = {t_short:g}, 10 time steps, dS = 1): max gamma error on S in [80, 120] is "
         f"{err_plain:.4f} with plain Crank-Nicolson and {err_smooth:.5f} with Rannacher smoothing "
         f"({err_plain / err_smooth:.0f}x smaller).")
    emit()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, key, getter in ((ax1, "delta", "delta_grid"), (ax2, "gamma", "gamma_grid")):
        ax.plot(S, exact[key], color=COLORS["exact"], lw=2.5, alpha=0.35, label="analytic")
        ax.plot(plain.S, getattr(plain, getter)(), ".-", color=COLORS["cn"], lw=0.8, ms=3, label="Crank-Nicolson")
        ax.plot(smooth.S, getattr(smooth, getter)(), "-", color=COLORS["cn_r"], lw=1.2,
                label="Crank-Nicolson + Rannacher")
        ax.set_xlim(85, 115)
        ax.set_xlabel("stock price S")
        ax.set_title(f"{key.capitalize()}, T = {t_short:g}, 10 time steps")
    ax2.set_ylim(-0.02, 0.11)
    ax1.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "greeks_rannacher.png")
    plt.close(fig)


if __name__ == "__main__":
    t_start = time.perf_counter()
    emit("# Results")
    emit()
    emit("Generated by experiments/run_experiments.py. Seed = 42.")
    emit()
    benchmark()
    convergence()
    variance_reduction()
    american()
    greeks()
    (RES / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\nDone in {time.perf_counter() - t_start:.0f} s. Figures in figures/, tables in results/results.md")
