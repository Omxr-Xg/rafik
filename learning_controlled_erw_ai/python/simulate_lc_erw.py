#!/usr/bin/env python3

"""Python fallback simulations for the Learning-Controlled ERW.

This mirrors the R scripts and is included because Rscript may be unavailable on
some systems. It generates CSV summaries and QQ-plot PDFs using numpy,
matplotlib and scipy.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"


def logistic_link(theta: float, eps: float = 0.05) -> float:
    return eps + (1 - 2 * eps) / (1 + math.exp(-theta))


def theta_for_p(p: float, eps: float = 0.05) -> float:
    if not eps < p < 1 - eps:
        raise ValueError("p must lie strictly between eps and 1 - eps")
    x = (p - eps) / (1 - 2 * eps)
    return math.log(x / (1 - x))


def clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def validate_simulation_args(n: int, reps: int | None = None) -> None:
    if not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer greater than or equal to 2")
    if reps is not None and (not isinstance(reps, int) or reps < 2):
        raise ValueError("reps must be an integer greater than or equal to 2")


@dataclass(frozen=True)
class Policy:
    name: str
    p_limit: float

    def get_p(self, t: int, s_t: int) -> float:
        if self.name == "fixed":
            return 0.60

        if self.name == "annealing":
            p_star = 0.60
            c = 0.10
            rho = 0.60
            return clip(p_star + c / ((t + 1) ** rho), 0.05, 0.72)

        if self.name == "logistic_learning":
            eps = 0.05
            p_star = 0.61
            theta_star = theta_for_p(p_star, eps)
            c = 0.45
            rho = 0.55
            theta_t = theta_star + c / ((t + 1) ** rho)
            return clip(logistic_link(theta_t, eps), 0.05, 0.72)

        if self.name == "damped_empirical":
            eps = 0.05
            p_star = 0.58
            theta_star = theta_for_p(p_star, eps)
            theta_lo = theta_for_p(0.06, eps)
            theta_hi = theta_for_p(0.72, eps)
            c = 0.35
            beta = 0.50
            rho = 0.60
            delta = 0.40
            theta_t = theta_star + c / ((t + 1) ** rho) - beta * s_t / (t ** (1 + delta))
            return logistic_link(clip(theta_t, theta_lo, theta_hi), eps)

        if self.name == "critical":
            return clip(0.75 - 0.12 / ((t + 1) ** 0.55), 0.05, 0.75)

        if self.name == "superdiffusive":
            return 0.80

        if self.name == "slow_variation":
            a_star = 0.20
            c = 0.12
            a_t = a_star + c / math.log(t + 2)
            return clip((1 + a_t) / 2, 0.05, 0.72)

        raise ValueError(f"Unknown policy: {self.name}")


def make_policy(name: str) -> Policy:
    limits = {
        "fixed": 0.60,
        "annealing": 0.60,
        "logistic_learning": 0.61,
        "damped_empirical": 0.58,
        "critical": 0.75,
        "superdiffusive": 0.80,
        "slow_variation": 0.60,
    }
    if name not in limits:
        raise ValueError(f"Unknown policy: {name}")
    return Policy(name=name, p_limit=limits[name])


def simulate_path(n: int, policy: Policy, rng: np.random.Generator, q: float = 0.5) -> tuple[float, float]:
    validate_simulation_args(n)
    if not 0 <= q <= 1:
        raise ValueError("q must be a probability in [0, 1]")

    x = np.empty(n, dtype=np.int8)
    s = np.empty(n, dtype=np.int32)
    p_tail = []

    x[0] = 1 if rng.random() < q else -1
    s[0] = int(x[0])

    for t in range(1, n):
        p_t = policy.get_p(t, int(s[t - 1]))
        if t > n - 101:
            p_tail.append(p_t)
        k = rng.integers(0, t)
        x[t] = x[k] if rng.random() < p_t else -x[k]
        s[t] = s[t - 1] + int(x[t])

    return float(s[-1] / math.sqrt(n)), float(np.mean(p_tail))


def simulate_full_path(n: int, policy: Policy, rng: np.random.Generator, q: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_simulation_args(n)
    x = np.empty(n, dtype=np.int8)
    s = np.empty(n, dtype=np.int32)
    p_hist = np.empty(n - 1)
    a_product = np.empty(n)

    x[0] = 1 if rng.random() < q else -1
    s[0] = int(x[0])
    a_product[0] = 1.0

    for t in range(1, n):
        p_t = policy.get_p(t, int(s[t - 1]))
        p_hist[t - 1] = p_t
        a_coeff = 2 * p_t - 1
        a_product[t] = a_product[t - 1] * (1 + a_coeff / t)
        k = rng.integers(0, t)
        x[t] = x[k] if rng.random() < p_t else -x[k]
        s[t] = s[t - 1] + int(x[t])

    return s, p_hist, a_product


def simulate_values(policy_name: str, n: int, reps: int, seed: int) -> tuple[Policy, np.ndarray, np.ndarray]:
    validate_simulation_args(n, reps)
    rng = np.random.default_rng(seed)
    policy = make_policy(policy_name)
    scaled = np.empty(reps)
    tail_p = np.empty(reps)
    for r in range(reps):
        scaled[r], tail_p[r] = simulate_path(n, policy, rng)
    return policy, scaled, tail_p


def summarize(policy_name: str, n: int, reps: int, seed: int) -> dict[str, float | int | str]:
    validate_simulation_args(n, reps)
    policy, scaled, tail_p = simulate_values(policy_name, n, reps, seed)
    a_limit = 2 * policy.p_limit - 1
    theoretical_variance = 1 / (1 - 2 * a_limit)
    empirical_mean = float(np.mean(scaled))
    empirical_variance = float(np.var(scaled, ddof=1))
    empirical_sd = math.sqrt(empirical_variance)
    variance_se = empirical_variance * math.sqrt(2 / (reps - 1))
    skewness = float(stats.skew(scaled, bias=False))
    excess_kurtosis = float(stats.kurtosis(scaled, fisher=True, bias=False))
    ks = stats.kstest(scaled, "norm", args=(0, math.sqrt(theoretical_variance)))

    return {
        "policy": policy_name,
        "n": n,
        "reps": reps,
        "limiting_p": policy.p_limit,
        "limiting_a": a_limit,
        "mean_tail_p": float(np.mean(tail_p)),
        "theoretical_variance": theoretical_variance,
        "empirical_mean": empirical_mean,
        "mean_se": empirical_sd / math.sqrt(reps),
        "empirical_variance": empirical_variance,
        "variance_se": variance_se,
        "variance_ci_low": empirical_variance - 1.96 * variance_se,
        "variance_ci_high": empirical_variance + 1.96 * variance_se,
        "variance_ratio": empirical_variance / theoretical_variance,
        "q025": float(np.quantile(scaled, 0.025)),
        "q500": float(np.quantile(scaled, 0.500)),
        "q975": float(np.quantile(scaled, 0.975)),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_qq_plot(policy_name: str, n: int, reps: int, seed: int) -> None:
    policy, scaled, _ = simulate_values(policy_name, n, reps, seed)
    a_limit = 2 * policy.p_limit - 1
    theoretical_sd = math.sqrt(1 / (1 - 2 * a_limit))
    osm = stats.norm.ppf((np.arange(1, reps + 1) - 0.5) / reps, scale=theoretical_sd)
    osr = np.sort(scaled)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(osm, osr, s=10, alpha=0.65)
    low = min(float(osm.min()), float(osr.min()))
    high = max(float(osm.max()), float(osr.max()))
    plt.plot([low, high], [low, high], color="red", linewidth=1)
    plt.title(f"LC-ERW QQ plot: {policy_name}")
    plt.xlabel("Theoretical Gaussian quantiles")
    plt.ylabel("Empirical quantiles")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"qq_{policy_name}_n{n}.pdf")
    plt.close()


def save_histogram_overlay(policy_name: str, n: int, reps: int, seed: int) -> None:
    policy, scaled, _ = simulate_values(policy_name, n, reps, seed)
    a_limit = 2 * policy.p_limit - 1
    theoretical_sd = math.sqrt(1 / (1 - 2 * a_limit))
    x_grid = np.linspace(float(scaled.min()) - 0.5, float(scaled.max()) + 0.5, 300)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 4.5))
    plt.hist(scaled, bins=32, density=True, alpha=0.55, label="Simulated")
    plt.plot(x_grid, stats.norm.pdf(x_grid, loc=0, scale=theoretical_sd), color="red", linewidth=2, label="Theory")
    plt.title(f"Scaled LC-ERW distribution: {policy_name}")
    plt.xlabel(r"$S_n/\sqrt{n}$")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"hist_{policy_name}_n{n}.pdf")
    plt.close()


def save_variance_ratio_plot(grid_rows: list[dict[str, float | int | str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    policies = ["fixed", "annealing", "logistic_learning", "damped_empirical"]
    labels = {
        "fixed": "Fixed",
        "annealing": "Annealing",
        "logistic_learning": "Logistic learning",
        "damped_empirical": "Projected empirical-risk",
    }

    plt.figure(figsize=(6.8, 4.6))
    for policy in policies:
        rows = [row for row in grid_rows if row["policy"] == policy]
        rows.sort(key=lambda row: int(row["n"]))
        n_values = [int(row["n"]) for row in rows]
        ratios = [float(row["variance_ratio"]) for row in rows]
        plt.plot(n_values, ratios, marker="o", linewidth=1.8, label=labels[policy])

    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("n")
    plt.ylabel("Empirical variance / theoretical variance")
    plt.title("Variance-ratio convergence across sample sizes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "variance_ratio_grid.pdf")
    plt.close()


def save_policy_trajectory_plot(n: int = 2500, seed: int = 321) -> None:
    rng = np.random.default_rng(seed)
    policies = ["fixed", "annealing", "damped_empirical", "slow_variation", "critical", "superdiffusive"]
    labels = {
        "fixed": "Fixed",
        "annealing": "Annealing",
        "damped_empirical": "Damped empirical",
        "slow_variation": "Slow variation",
        "critical": "Critical",
        "superdiffusive": "Superdiffusive",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.6))
    for name in policies:
        _, p_hist, _ = simulate_full_path(n, make_policy(name), rng)
        a_hist = 2 * p_hist - 1
        plt.plot(np.arange(1, n), a_hist, linewidth=1.3, label=labels[name])
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xlabel("n")
    plt.ylabel(r"learned coefficient $a_n$")
    plt.title("Representative learned-memory trajectories")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "policy_trajectories.pdf")
    plt.close()


def compute_fclt_covariance(n: int = 2500, reps: int = 400, seed: int = 4321) -> list[dict[str, float | int | str]]:
    validate_simulation_args(n, reps)
    rng = np.random.default_rng(seed)
    policy = make_policy("fixed")
    a = 2 * policy.p_limit - 1
    times = np.array([0.25, 0.50, 0.75, 1.00])
    indices = np.maximum(1, (n * times).astype(int)) - 1
    values = np.empty((reps, len(times)))
    for r in range(reps):
        s, _, _ = simulate_full_path(n, policy, rng)
        values[r, :] = s[indices] / math.sqrt(n)
    empirical = np.cov(values, rowvar=False, ddof=1)
    rows: list[dict[str, float | int | str]] = []
    for i, s_time in enumerate(times):
        for j, t_time in enumerate(times):
            lower = min(float(s_time), float(t_time))
            upper = max(float(s_time), float(t_time))
            theory = (lower ** (1 - a)) * (upper ** a) / (1 - 2 * a)
            rows.append({
                "policy": "fixed",
                "n": n,
                "reps": reps,
                "s": float(s_time),
                "t": float(t_time),
                "empirical_covariance": float(empirical[i, j]),
                "theoretical_covariance": float(theory),
                "absolute_error": float(abs(empirical[i, j] - theory)),
            })
    return rows


def save_extended_diagnostics(n: int = 2500, reps: int = 500, seed: int = 5432) -> None:
    validate_simulation_args(n, reps)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    start = time.perf_counter()

    critical_vals = np.empty(reps)
    super_vals = np.empty(reps)
    slow_ratios = np.empty(reps)
    for r in range(reps):
        s_crit, _, _ = simulate_full_path(n, make_policy("critical"), rng)
        critical_vals[r] = s_crit[-1] / math.sqrt(n * math.log(n))

        s_super, _, a_product = simulate_full_path(n, make_policy("superdiffusive"), rng)
        a_super = 2 * make_policy("superdiffusive").p_limit - 1
        super_vals[r] = s_super[-1] / (n ** a_super)

        _, _, slow_product = simulate_full_path(n, make_policy("slow_variation"), rng)
        a_slow = 2 * make_policy("slow_variation").p_limit - 1
        c = 0.12
        slow_ratios[r] = slow_product[-1] / ((n ** a_slow) * (math.log(n) ** c))

    elapsed = time.perf_counter() - start
    rows.extend([
        {
            "diagnostic": "critical_variance",
            "n": n,
            "reps": reps,
            "value": float(np.var(critical_vals, ddof=1)),
            "target": 1.0,
            "runtime_seconds": elapsed,
        },
        {
            "diagnostic": "superdiffusive_scaled_sd",
            "n": n,
            "reps": reps,
            "value": float(np.std(super_vals, ddof=1)),
            "target": "finite non-Gaussian terminal law",
            "runtime_seconds": elapsed,
        },
        {
            "diagnostic": "slow_variation_product_ratio_mean",
            "n": n,
            "reps": reps,
            "value": float(np.mean(slow_ratios)),
            "target": "finite positive constant",
            "runtime_seconds": elapsed,
        },
    ])
    write_csv(RESULTS_DIR / "extended_diagnostics_python.csv", rows)

    plt.figure(figsize=(7.0, 4.5))
    plt.hist(critical_vals, bins=30, density=True, alpha=0.50, label=r"critical $S_n/\sqrt{n\log n}$")
    x_grid = np.linspace(float(critical_vals.min()) - 0.5, float(critical_vals.max()) + 0.5, 300)
    plt.plot(x_grid, stats.norm.pdf(x_grid), color="red", linewidth=2, label="N(0,1)")
    plt.xlabel("scaled value")
    plt.ylabel("density")
    plt.title("Critical scaling diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "critical_scaling_diagnostic.pdf")
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    plt.hist(super_vals, bins=30, density=True, alpha=0.60)
    plt.xlabel(r"$S_n/n^a$")
    plt.ylabel("density")
    plt.title("Superdiffusive terminal scaling diagnostic")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "superdiffusive_scaling_diagnostic.pdf")
    plt.close()


def main() -> None:
    policies = ["fixed", "annealing", "logistic_learning", "damped_empirical"]

    summary_rows = [
        summarize(policy, n=2500, reps=800, seed=123 + idx)
        for idx, policy in enumerate(policies, start=1)
    ]
    write_csv(RESULTS_DIR / "lc_erw_simulation_summary_python.csv", summary_rows)

    grid_rows = []
    idx = 1
    for policy in policies:
        for n in [1000, 2500, 5000]:
            grid_rows.append(summarize(policy, n=n, reps=500, seed=7000 + idx))
            idx += 1
    write_csv(RESULTS_DIR / "lc_erw_variance_grid_python.csv", grid_rows)
    save_variance_ratio_plot(grid_rows)
    write_csv(RESULTS_DIR / "fclt_covariance_python.csv", compute_fclt_covariance())
    save_policy_trajectory_plot()
    save_extended_diagnostics()

    for idx, policy in enumerate(policies, start=1):
        save_qq_plot(policy, n=2500, reps=800, seed=900 + idx)
    for idx, policy in enumerate(policies, start=1):
        save_histogram_overlay(policy, n=2500, reps=800, seed=1200 + idx)

    print("Wrote simulation summaries and plots to", RESULTS_DIR)
    for row in summary_rows:
        print(
            row["policy"],
            "var_ratio=",
            f"{row['variance_ratio']:.3f}",
            "ks_p=",
            f"{row['ks_p_value']:.3f}",
        )
    print("Wrote FCLT, critical, superdiffusive and slow-variation diagnostics")


if __name__ == "__main__":
    main()
