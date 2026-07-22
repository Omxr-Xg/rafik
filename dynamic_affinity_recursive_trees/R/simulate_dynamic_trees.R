#!/usr/bin/env Rscript
# Base-R mirror of the Python Monte Carlo pipeline.
# Prefer python/simulate_dynamic_trees.py if Rscript is unavailable.

set.seed(20260718)

V_lambda <- function(lam) (2 + 8 * lam - 2 * lam^2) / (lam + 1)^3
a_lambda <- function(lam) 2 / (1 + lam)

sample_parent <- function(n, lam) {
  u <- runif(1)
  target <- u * n * (n - 1)
  if (abs(lam - 1) < 1e-12) {
    return(min(max(floor(u * (n - 1)) + 1L, 1L), n - 1L))
  }
  a <- 1 - lam
  b <- lam * n + 1 - lam
  if (abs(a) < 1e-12) {
    x <- target / b
  } else {
    disc <- max(b * b + 4 * a * target, 0)
    x <- (-b + sqrt(disc)) / (2 * a)
  }
  k <- as.integer(ceiling(x - 1e-12))
  min(max(k, 1L), n - 1L)
}

simulate_final_depth <- function(n, lam, n_rep) {
  depths <- numeric(n_rep)
  parents <- numeric(n_rep)
  for (r in seq_len(n_rep)) {
    depth <- integer(n)
    depth[1] <- 0L
    parent_n <- 1L
    for (m in 2:n) {
      parent <- sample_parent(m, lam)
      depth[m] <- depth[parent] + 1L
      if (m == n) parent_n <- parent
    }
    depths[r] <- depth[n]
    parents[r] <- parent_n / n
  }
  list(depths = depths, parents = parents)
}

root <- normalizePath(file.path(".."), mustWork = FALSE)
if (!dir.exists(file.path(root, "results"))) {
  root <- getwd()
}
results <- file.path(root, "results")
dir.create(results, showWarnings = FALSE, recursive = TRUE)

message("R simulations write to ", results)
message("For full figure suite, run python/simulate_dynamic_trees.py")

out <- simulate_final_depth(500L, 1.0, 200L)
write.csv(
  data.frame(
    n = 500,
    lambda = 1,
    emp_mean = mean(out$depths),
    emp_var = var(out$depths)
  ),
  file.path(results, "r_smoke_test.csv"),
  row.names = FALSE
)
message("Wrote results/r_smoke_test.csv")
