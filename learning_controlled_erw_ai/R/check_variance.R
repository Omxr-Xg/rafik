#!/usr/bin/env Rscript

# Variance convergence checks for the LC-ERW central limit theorem.

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg) == 1) sub("^--file=", "", script_arg) else getwd()
script_dir <- dirname(normalizePath(script_path))
project_dir <- dirname(script_dir)

source(file.path(script_dir, "simulate_lc_erw.R"))

run_variance_grid <- function() {
  policies <- c("fixed", "annealing", "logistic_learning", "damped_empirical")
  sizes <- c(1000, 2500, 5000)
  reps <- 500

  rows <- list()
  idx <- 1
  for (policy in policies) {
    for (n in sizes) {
      rows[[idx]] <- simulate_policy(policy, n = n, reps = reps, seed = 7000 + idx)
      idx <- idx + 1
    }
  }

  results <- do.call(rbind, rows)
  print(results[, c(
    "policy", "n", "reps", "theoretical_variance",
    "empirical_variance", "variance_ci_low", "variance_ci_high",
    "variance_ratio", "ks_p_value"
  )], digits = 4)

  out_dir <- file.path(project_dir, "results")
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }
  write.csv(results, file.path(out_dir, "lc_erw_variance_grid.csv"), row.names = FALSE)
  invisible(results)
}

if (sys.nframe() == 0) {
  run_variance_grid()
}
