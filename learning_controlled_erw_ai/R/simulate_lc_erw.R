#!/usr/bin/env Rscript

# Reproducible simulations for the Learning-Controlled Elephant Random Walk.
# The code uses base R only.

logistic_link <- function(theta, eps = 0.05) {
  eps + (1 - 2 * eps) / (1 + exp(-theta))
}

theta_for_p <- function(p, eps = 0.05) {
  if (p <= eps || p >= 1 - eps) {
    stop("p must lie strictly between eps and 1 - eps")
  }
  qlogis((p - eps) / (1 - 2 * eps))
}

clip <- function(x, lo, hi) {
  pmin(pmax(x, lo), hi)
}

validate_simulation_args <- function(n, reps = NULL) {
  if (!is.numeric(n) || length(n) != 1 || is.na(n) || n < 2 || n != as.integer(n)) {
    stop("n must be a single integer greater than or equal to 2")
  }
  if (!is.null(reps) &&
      (!is.numeric(reps) || length(reps) != 1 || is.na(reps) || reps < 2 || reps != as.integer(reps))) {
    stop("reps must be a single integer greater than or equal to 2")
  }
}

make_policy <- function(name) {
  if (name == "fixed") {
    p_star <- 0.60
    return(list(
      name = name,
      p_limit = p_star,
      theta0 = NA_real_,
      get_p = function(t, S_t, theta) p_star,
      update = function(t, S_next, theta) theta
    ))
  }

  if (name == "annealing") {
    p_star <- 0.60
    c <- 0.10
    rho <- 0.60
    return(list(
      name = name,
      p_limit = p_star,
      theta0 = NA_real_,
      get_p = function(t, S_t, theta) clip(p_star + c / (t + 1)^rho, 0.05, 0.72),
      update = function(t, S_next, theta) theta
    ))
  }

  if (name == "logistic_learning") {
    eps <- 0.05
    p_star <- 0.61
    theta_star <- theta_for_p(p_star, eps)
    c <- 0.45
    rho <- 0.55
    return(list(
      name = name,
      p_limit = p_star,
      theta0 = theta_star + c,
      get_p = function(t, S_t, theta) {
        theta_t <- theta_star + c / (t + 1)^rho
        clip(logistic_link(theta_t, eps), 0.05, 0.72)
      },
      update = function(t, S_next, theta) theta
    ))
  }

  if (name == "damped_empirical") {
    eps <- 0.05
    p_star <- 0.58
    theta_star <- theta_for_p(p_star, eps)
    theta_lo <- theta_for_p(0.06, eps)
    theta_hi <- theta_for_p(0.72, eps)
    c <- 0.35
    beta <- 0.50
    rho <- 0.60
    delta <- 0.40
    return(list(
      name = name,
      p_limit = p_star,
      theta0 = theta_star,
      get_p = function(t, S_t, theta) {
        theta_t <- theta_star + c / (t + 1)^rho - beta * S_t / t^(1 + delta)
        theta_t <- clip(theta_t, theta_lo, theta_hi)
        logistic_link(theta_t, eps)
      },
      update = function(t, S_next, theta) theta
    ))
  }

  stop("Unknown policy: ", name)
}

simulate_lc_erw_path <- function(n, policy, q = 0.5) {
  validate_simulation_args(n)
  if (!is.numeric(q) || length(q) != 1 || is.na(q) || q < 0 || q > 1) {
    stop("q must be a probability in [0, 1]")
  }

  X <- integer(n)
  S <- integer(n)
  p_hist <- numeric(n - 1)
  theta <- policy$theta0

  X[1] <- if (runif(1) < q) 1L else -1L
  S[1] <- X[1]

  for (t in seq_len(n - 1)) {
    p_t <- policy$get_p(t, S[t], theta)
    p_hist[t] <- p_t
    k <- sample.int(t, 1)
    X[t + 1] <- if (runif(1) < p_t) X[k] else -X[k]
    S[t + 1] <- S[t] + X[t + 1]
    theta <- policy$update(t, S[t + 1], theta)
  }

  list(S_n = S[n], scaled = S[n] / sqrt(n), mean_p_tail = mean(tail(p_hist, 100)))
}

simulate_scaled_values <- function(policy_name, n = 2500, reps = 800, seed = 123) {
  validate_simulation_args(n, reps)
  set.seed(seed)
  policy <- make_policy(policy_name)
  scaled <- numeric(reps)
  tail_p <- numeric(reps)

  for (r in seq_len(reps)) {
    path <- simulate_lc_erw_path(n, policy)
    scaled[r] <- path$scaled
    tail_p[r] <- path$mean_p_tail
  }

  list(policy = policy, scaled = scaled, tail_p = tail_p)
}

summarise_scaled_values <- function(policy_name, n, reps, scaled, tail_p) {
  validate_simulation_args(n, reps)
  policy <- make_policy(policy_name)
  a_limit <- 2 * policy$p_limit - 1
  theoretical_variance <- 1 / (1 - 2 * a_limit)
  empirical_variance <- var(scaled)
  empirical_mean <- mean(scaled)
  empirical_sd <- sd(scaled)
  variance_se <- empirical_variance * sqrt(2 / (reps - 1))
  variance_ci_low <- empirical_variance - 1.96 * variance_se
  variance_ci_high <- empirical_variance + 1.96 * variance_se
  skewness <- mean((scaled - empirical_mean)^3) / empirical_sd^3
  excess_kurtosis <- mean((scaled - empirical_mean)^4) / empirical_sd^4 - 3
  ks <- suppressWarnings(ks.test(scaled, "pnorm", mean = 0, sd = sqrt(theoretical_variance)))

  data.frame(
    policy = policy_name,
    n = n,
    reps = reps,
    limiting_p = policy$p_limit,
    limiting_a = a_limit,
    mean_tail_p = mean(tail_p),
    theoretical_variance = theoretical_variance,
    empirical_mean = empirical_mean,
    mean_se = empirical_sd / sqrt(reps),
    empirical_variance = empirical_variance,
    variance_se = variance_se,
    variance_ci_low = variance_ci_low,
    variance_ci_high = variance_ci_high,
    variance_ratio = empirical_variance / theoretical_variance,
    q025 = unname(quantile(scaled, 0.025)),
    q500 = unname(quantile(scaled, 0.500)),
    q975 = unname(quantile(scaled, 0.975)),
    skewness = skewness,
    excess_kurtosis = excess_kurtosis,
    ks_statistic = unname(ks$statistic),
    ks_p_value = ks$p.value,
    row.names = NULL
  )
}

simulate_policy <- function(policy_name, n = 2500, reps = 800, seed = 123) {
  values <- simulate_scaled_values(policy_name, n = n, reps = reps, seed = seed)
  summarise_scaled_values(policy_name, n, reps, values$scaled, values$tail_p)
}

save_qq_plot <- function(policy_name, n = 2500, reps = 800, seed = 123, out_dir = "results") {
  values <- simulate_scaled_values(policy_name, n = n, reps = reps, seed = seed)
  a_limit <- 2 * values$policy$p_limit - 1
  theoretical_sd <- sqrt(1 / (1 - 2 * a_limit))

  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }

  output <- file.path(out_dir, paste0("qq_", policy_name, "_n", n, ".pdf"))
  grDevices::pdf(output, width = 6, height = 6)
  qqnorm(values$scaled, main = paste("LC-ERW QQ plot:", policy_name))
  qqline(values$scaled, distribution = function(p) qnorm(p, sd = theoretical_sd), col = 2)
  grDevices::dev.off()
  invisible(output)
}

run_default_experiment <- function() {
  policies <- c("fixed", "annealing", "logistic_learning", "damped_empirical")
  results <- do.call(rbind, lapply(seq_along(policies), function(i) {
    simulate_policy(policies[i], seed = 123 + i)
  }))

  print(results, digits = 4)

  script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  script_path <- if (length(script_arg) == 1) sub("^--file=", "", script_arg) else getwd()
  out_dir <- file.path(dirname(dirname(normalizePath(script_path))), "results")
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }
  write.csv(results, file.path(out_dir, "lc_erw_simulation_summary.csv"), row.names = FALSE)
  invisible(lapply(seq_along(policies), function(i) {
    save_qq_plot(policies[i], seed = 900 + i, out_dir = out_dir)
  }))
  invisible(results)
}

if (sys.nframe() == 0) {
  run_default_experiment()
}
