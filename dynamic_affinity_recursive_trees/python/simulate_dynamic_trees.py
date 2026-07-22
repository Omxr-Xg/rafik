#!/usr/bin/env python3
"""
Simulations for preferential recursive trees with dynamic affinities.

Attachment probabilities for node n (n >= 2):
    P(parent = k) = (Lambda * n + 2*(1-Lambda)*k) / (n*(n-1)),  k = 1..n-1

Theory (age index Lambda in [0, 2]):
    E[D_n] ~ (2/(1+Lambda)) * H_{n-1} + C_Lambda
    Var[D_n] ~ V_Lambda * H_{n-1},
        V_Lambda = (2 + 8*Lambda - 2*Lambda^2)/(Lambda+1)^3
    (D_n - E[D_n]) / sqrt(ln n) -> N(0, V_Lambda)
    D_n / ln n -> 2/(1+Lambda) in probability
    J_n / n -> density f(x) = Lambda + 2*(1-Lambda)*x on (0,1)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260718)


def V_lambda(lam: float) -> float:
    return (2.0 + 8.0 * lam - 2.0 * lam**2) / (lam + 1.0) ** 3


def a_lambda(lam: float) -> float:
    return 2.0 / (1.0 + lam)


def harmonic(n: int) -> float:
    if n <= 0:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, n + 1)))


def harmonic2(n: int) -> float:
    if n <= 0:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, n + 1) ** 2))


def sample_parent(n: int, lam: float, rng: np.random.Generator) -> int:
    """
    O(1) inverse-CDF sample of parent k in {1,...,n-1}.

    Cumulative mass:
        F(k) = (Lambda*n*k + (1-Lambda)*k*(k+1)) / (n*(n-1)).
    Return the smallest integer k with F(k) >= U.
    """
    u = float(rng.random())
    target = u * n * (n - 1)
    if abs(lam - 1.0) < 1e-12:
        k = int(math.floor(u * (n - 1))) + 1
        return min(max(k, 1), n - 1)

    # Solve F(x) = u for real x, then take ceil.
    # (1-Lambda) x^2 + (Lambda*n + 1 - Lambda) x - target = 0
    a = 1.0 - lam
    b = lam * n + 1.0 - lam
    if abs(a) < 1e-12:
        x = target / b
    else:
        disc = max(b * b + 4.0 * a * target, 0.0)
        # For a>0 take positive root; for a<0 take the root in (0,n).
        x = (-b + math.sqrt(disc)) / (2.0 * a)
    k = int(math.ceil(x - 1e-12))
    return min(max(k, 1), n - 1)


def simulate_tree_depths(
    n: int, lam: float, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Grow one tree of size n. Return depths[0..n] (index 0 unused) and J_n."""
    depth = np.zeros(n + 1, dtype=np.int32)
    parent_n = 1
    for m in range(2, n + 1):
        parent = sample_parent(m, lam, rng)
        depth[m] = depth[parent] + 1
        if m == n:
            parent_n = parent
    return depth, parent_n


def simulate_final_depth(
    n: int, lam: float, n_rep: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    depths = np.empty(n_rep, dtype=np.float64)
    parents = np.empty(n_rep, dtype=np.float64)
    for i in range(n_rep):
        depth, parent = simulate_tree_depths(n, lam, rng)
        depths[i] = depth[n]
        parents[i] = parent / float(n)
    return depths, parents


def exact_mean_special(n: int, lam: float) -> float:
    """Closed-form means for Lambda in {0,1,2}; else asymptotic proxy."""
    if n <= 1:
        return 0.0
    if abs(lam - 0.0) < 1e-12:
        return 2.0 * (harmonic(n) - 1.0) if n >= 3 else 1.0
    if abs(lam - 1.0) < 1e-12:
        return harmonic(n - 1)
    if abs(lam - 2.0) < 1e-12:
        return (2.0 / 3.0) * harmonic(n - 1) + 1.0 / 3.0
    return a_lambda(lam) * harmonic(n - 1)


def exact_var_special(n: int, lam: float) -> float:
    """Closed-form variances for Lambda in {0,1,2}; else V_Lambda H_{n-1}."""
    if n <= 2:
        return 0.0
    if abs(lam - 0.0) < 1e-12:
        return 2.0 * harmonic(n) - 4.0 * harmonic2(n) + 2.0
    if abs(lam - 1.0) < 1e-12:
        return harmonic(n - 1) - harmonic2(n - 1)
    if abs(lam - 2.0) < 1e-12:
        return (
            (10.0 / 27.0) * harmonic(n - 1)
            + (12.0 / 27.0) * harmonic2(n)
            - 28.0 / 27.0
            + 12.0 / (27.0 * n)
        )
    return V_lambda(lam) * harmonic(n - 1)


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


COLORS = {0.0: "#1f77b4", 1.0: "#ff7f0e", 2.0: "#2ca02c"}


def plot_mean_variance_checks(summary_rows: list[dict]) -> None:
    """Compare Monte Carlo moments to exact finite-n formulae (same color)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for lam in sorted({float(r["lambda"]) for r in summary_rows}):
        rows = [r for r in summary_rows if float(r["lambda"]) == lam]
        ns = np.array([int(r["n"]) for r in rows])
        emp_mean = np.array([float(r["emp_mean"]) for r in rows])
        thy_mean = np.array([float(r["thy_mean"]) for r in rows])
        emp_var = np.array([float(r["emp_var"]) for r in rows])
        thy_var = np.array([float(r["thy_var"]) for r in rows])
        c = COLORS.get(lam, "black")
        axes[0].plot(ns, emp_mean, "o", color=c, label=rf"sim $\Lambda={lam:g}$")
        axes[0].plot(ns, thy_mean, "--", color=c, label=rf"exact $\Lambda={lam:g}$")
        axes[1].plot(ns, emp_var, "o", color=c, label=rf"sim $\Lambda={lam:g}$")
        axes[1].plot(ns, thy_var, "--", color=c, label=rf"exact $\Lambda={lam:g}$")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel(r"$n$")
    axes[1].set_xlabel(r"$n$")
    axes[0].set_ylabel(r"$\mathbb{E}[D_n]$")
    axes[1].set_ylabel(r"$\mathrm{Var}[D_n]$")
    axes[0].set_title("Mean depth (exact formulae)")
    axes[1].set_title("Variance (exact formulae)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "mean_variance_check.pdf")
    fig.savefig(RESULTS / "mean_variance_check.png", dpi=150)
    plt.close(fig)


def plot_lln(summary_rows: list[dict]) -> None:
    """
    Compare emp mean / ln n to exact E[D_n]/ln n (not only the n->inf limit).

    The horizontal asymptotes 2/(1+Lambda) are approached slowly because
    E[D_n] = a H_{n-1} + C + o(1), so E[D_n]/ln n = a (H_{n-1}/ln n) + C/ln n.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for lam in sorted({float(r["lambda"]) for r in summary_rows}):
        rows = [r for r in summary_rows if float(r["lambda"]) == lam]
        ns = np.array([int(r["n"]) for r in rows])
        emp_ratio = np.array([float(r["emp_mean"]) / math.log(int(r["n"])) for r in rows])
        thy_ratio = np.array([float(r["thy_mean"]) / math.log(int(r["n"])) for r in rows])
        c = COLORS.get(lam, "black")
        ax.plot(ns, emp_ratio, "o", color=c, label=rf"sim $\Lambda={lam:g}$")
        ax.plot(ns, thy_ratio, "--", color=c, label=rf"exact $\Lambda={lam:g}$")
        ax.axhline(a_lambda(lam), color=c, linestyle=":", alpha=0.55, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\mathbb{E}[D_n]/\ln n$")
    ax.set_title(r"Finite-$n$ mean ratio (dotted = asymptotic limit $2/(1+\Lambda)$)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "lln_convergence.pdf")
    fig.savefig(RESULTS / "lln_convergence.png", dpi=150)
    plt.close(fig)


def plot_variance_ratio(summary_rows: list[dict]) -> None:
    """Show Var[D_n]/H_{n-1} approaching V_Lambda (slow O(1/ln n) rate)."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for lam in sorted({float(r["lambda"]) for r in summary_rows}):
        rows = [r for r in summary_rows if float(r["lambda"]) == lam]
        ns = np.array([int(r["n"]) for r in rows])
        emp = np.array(
            [
                float(r["emp_var"]) / harmonic(int(r["n"]) - 1)
                for r in rows
            ]
        )
        thy = np.array(
            [
                float(r["thy_var"]) / harmonic(int(r["n"]) - 1)
                for r in rows
            ]
        )
        c = COLORS.get(lam, "black")
        ax.plot(ns, emp, "o", color=c, label=rf"sim $\Lambda={lam:g}$")
        ax.plot(ns, thy, "--", color=c, label=rf"exact $\Lambda={lam:g}$")
        ax.axhline(V_lambda(lam), color=c, linestyle=":", alpha=0.55, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\mathrm{Var}[D_n]/H_{n-1}$")
    ax.set_title(r"Variance coefficient (dotted = asymptotic $V_\Lambda$)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "variance_ratio.pdf")
    fig.savefig(RESULTS / "variance_ratio.png", dpi=150)
    plt.close(fig)


def plot_clt_qq(depths: np.ndarray, n: int, lam: float, tag: str) -> None:
    """
    Studentized CLT diagnostics for integer-valued depths.

    D_n is discrete, so on the studentized scale the atoms sit on a lattice of
    spacing 1/sd. A fixed number of narrow bins then inflates density bar heights
    (empty bins between atoms), which wrongly looks like a mismatch with N(0,1)
    even when Var[(D-mu)/sd] is already ~1. We therefore use lattice-aligned bins
    of width 1/sd; then bar heights estimate the local density and match phi.
    """
    mu = exact_mean_special(n, lam)
    sd = math.sqrt(exact_var_special(n, lam))
    z = (depths - mu) / sd
    delta = 1.0 / sd  # lattice spacing of integer depths on the z-scale

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))

    # Lattice-aligned histogram: each bin covers one depth atom.
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    left = z_min - 0.5 * delta
    right = z_max + 0.5 * delta
    bins = np.arange(left, right + 0.5 * delta, delta)
    axes[0].hist(
        z,
        bins=bins,
        density=True,
        alpha=0.75,
        color="steelblue",
        edgecolor="white",
        label="simulation",
    )
    xs = np.linspace(z_min - 0.5, z_max + 0.5, 400)
    axes[0].plot(xs, stats.norm.pdf(xs, 0.0, 1.0), "r-", lw=2, label=r"$N(0,1)$")
    axes[0].set_title(rf"Studentized histogram ($\Lambda={lam:g}$, $n={n}$)")
    axes[0].set_xlabel(r"$(D_n-\mu_n)/\sqrt{\mathrm{Var}[D_n]}$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Continuity-corrected points are unused; staircase QQ is the honest discrete picture.
    (osm, osr), (slope, intercept, _r) = stats.probplot(z, dist="norm")
    axes[1].plot(osm, osr, "o", markersize=3, color="steelblue", alpha=0.7)
    axes[1].plot(osm, slope * osm + intercept, "r-", lw=1.5, label="least-squares fit")
    axes[1].plot(osm, osm, "k--", lw=1.0, alpha=0.7, label=r"$y=x$")
    axes[1].set_title(rf"QQ vs $N(0,1)$ ($\Lambda={lam:g}$)")
    axes[1].set_xlabel("Theoretical quantiles")
    axes[1].set_ylabel("Ordered values")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"clt_{tag}.pdf")
    fig.savefig(RESULTS / f"clt_{tag}.png", dpi=150)
    plt.close(fig)


def plot_parent_density(parents: np.ndarray, lam: float, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.hist(
        parents, bins=40, density=True, alpha=0.75, color="seagreen", edgecolor="white"
    )
    xs = np.linspace(0.001, 0.999, 400)
    dens = lam + 2.0 * (1.0 - lam) * xs
    ax.plot(xs, dens, "r-", lw=2, label=r"$f_\Lambda(x)=\Lambda+2(1-\Lambda)x$")
    ax.set_xlabel(r"$J_n/n$")
    ax.set_ylabel("density")
    ax.set_title(rf"Parent-index limit ($\Lambda={lam:g}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"parent_density_{tag}.pdf")
    fig.savefig(RESULTS / f"parent_density_{tag}.png", dpi=150)
    plt.close(fig)


def plot_v_lambda_curve() -> None:
    xs = np.linspace(0.0, 2.0, 400)
    ys = V_lambda(xs)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(xs, ys, "b-", lw=2)
    for lam, name in [(0.0, "young"), (1.0, "uniform"), (2.0, "old")]:
        ax.scatter([lam], [V_lambda(lam)], s=50, zorder=3)
        ax.annotate(
            rf"$V_{{{lam:g}}}={V_lambda(lam):.3f}$ ({name})",
            xy=(lam, V_lambda(lam)),
            xytext=(lam + 0.05, V_lambda(lam) + 0.12),
            fontsize=8,
        )
    ax.set_xlabel(r"$\Lambda$")
    ax.set_ylabel(r"$V_\Lambda$")
    ax.set_title(r"$V_\Lambda=(2+8\Lambda-2\Lambda^2)/(\Lambda+1)^3$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "v_lambda_curve.pdf")
    fig.savefig(RESULTS / "v_lambda_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("Dynamic-affinity recursive tree simulations")
    print(f"Results -> {RESULTS}")

    lambdas = [0.0, 1.0, 2.0]
    sizes = [200, 500, 1000, 2000, 5000]
    n_rep_grid = 800
    n_rep_clt = 2000
    n_clt = 5000

    summary_rows: list[dict] = []

    for lam in lambdas:
        for n in sizes:
            print(f"  grid: Lambda={lam}, n={n}, reps={n_rep_grid}")
            depths, parents = simulate_final_depth(n, lam, n_rep_grid, RNG)
            emp_mean = float(np.mean(depths))
            emp_var = float(np.var(depths, ddof=1))
            thy_mean = exact_mean_special(n, lam)
            thy_var = exact_var_special(n, lam)
            summary_rows.append(
                {
                    "lambda": lam,
                    "n": n,
                    "n_rep": n_rep_grid,
                    "emp_mean": emp_mean,
                    "thy_mean": thy_mean,
                    "emp_var": emp_var,
                    "thy_var": thy_var,
                    "emp_mean_over_logn": emp_mean / math.log(n),
                    "thy_mean_over_logn": thy_mean / math.log(n),
                    "limit_a": a_lambda(lam),
                    "emp_var_over_Hn": emp_var / harmonic(n - 1),
                    "thy_var_over_Hn": thy_var / harmonic(n - 1),
                    "V_lambda": V_lambda(lam),
                    "parent_mean": float(np.mean(parents)),
                }
            )

    save_csv(
        RESULTS / "depth_moments_summary.csv",
        summary_rows,
        [
            "lambda",
            "n",
            "n_rep",
            "emp_mean",
            "thy_mean",
            "emp_var",
            "thy_var",
            "emp_mean_over_logn",
            "thy_mean_over_logn",
            "limit_a",
            "emp_var_over_Hn",
            "thy_var_over_Hn",
            "V_lambda",
            "parent_mean",
        ],
    )

    plot_mean_variance_checks(summary_rows)
    plot_lln(summary_rows)
    plot_variance_ratio(summary_rows)
    plot_v_lambda_curve()

    clt_rows: list[dict] = []
    for lam, tag in [(0.0, "young"), (1.0, "uniform"), (2.0, "old")]:
        print(f"  CLT: Lambda={lam}, n={n_clt}, reps={n_rep_clt}")
        depths, parents = simulate_final_depth(n_clt, lam, n_rep_clt, RNG)
        plot_clt_qq(depths, n_clt, lam, tag)
        plot_parent_density(parents, lam, tag)
        mu = exact_mean_special(n_clt, lam)
        thy_var = exact_var_special(n_clt, lam)
        z = (depths - mu) / math.sqrt(thy_var)
        clt_rows.append(
            {
                "lambda": lam,
                "tag": tag,
                "n": n_clt,
                "n_rep": n_rep_clt,
                "emp_mean": float(np.mean(depths)),
                "thy_mean": mu,
                "emp_var": float(np.var(depths, ddof=1)),
                "thy_var": thy_var,
                "emp_student_var": float(np.var(z, ddof=1)),
                "V_lambda": V_lambda(lam),
                "var_over_logn": thy_var / math.log(n_clt),
                "ks_pvalue": float(stats.kstest(z, "norm").pvalue),
            }
        )

    save_csv(
        RESULTS / "clt_summary.csv",
        clt_rows,
        [
            "lambda",
            "tag",
            "n",
            "n_rep",
            "emp_mean",
            "thy_mean",
            "emp_var",
            "thy_var",
            "emp_student_var",
            "V_lambda",
            "var_over_logn",
            "ks_pvalue",
        ],
    )

    print("Done. Figures written to results/")


if __name__ == "__main__":
    main()
