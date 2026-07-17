#!/usr/bin/env Rscript

# Reproducible simulations for the Learning-Controlled Elephant Random Walk.
# The code uses base R only.

logistic_link <- function(theta, eps = 0.05) {
  eps + (1 - 2 * eps) / (1 + exp(-theta))
}

theta_for_p <- function(p, eps = 0.05) {
  qlogis((p - eps) / (1 - 2 * eps))
}

clip <- function(x, lo, hi) {
  pmin(pmax(x, lo), hi)
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

  if (name == "adaptive_controller") {
    eps <- 0.05
    p_star <- 0.58
    b <- theta_for_p(p_star, eps)
    lambda <- 0.20
    eta <- 0.75
    return(list(
      name = name,
      p_limit = p_star,
      theta0 = b + 0.30,
      get_p = function(t, S_t, theta) clip(logistic_link(theta, eps), 0.05, 0.72),
      update = function(t, S_next, theta) {
        gamma <- 0.8 / (t + 10)^eta
        theta + gamma * (b - theta - lambda * S_next / (t + 1))
      }
    ))
  }

  stop("Unknown policy: ", name)
}

simulate_lc_erw_path <- function(n, policy, q = 0.5) {
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

simulate_policy <- function(policy_name, n = 2500, reps = 800, seed = 123) {
  set.seed(seed)
  policy <- make_policy(policy_name)
  scaled <- numeric(reps)
  tail_p <- numeric(reps)

  for (r in seq_len(reps)) {
    path <- simulate_lc_erw_path(n, policy)
    scaled[r] <- path$scaled
    tail_p[r] <- path$mean_p_tail
  }

  a_limit <- 2 * policy$p_limit - 1
  theoretical_variance <- 1 / (1 - 2 * a_limit)
  empirical_variance <- var(scaled)
  empirical_mean <- mean(scaled)
  empirical_sd <- sd(scaled)
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
    empirical_variance = empirical_variance,
    variance_ratio = empirical_variance / theoretical_variance,
    skewness = skewness,
    excess_kurtosis = excess_kurtosis,
    ks_statistic = unname(ks$statistic),
    ks_p_value = ks$p.value,
    row.names = NULL
  )
}

run_default_experiment <- function() {
  policies <- c("fixed", "annealing", "logistic_learning", "adaptive_controller")
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
  invisible(results)
}

if (sys.nframe() == 0) {
  run_default_experiment()
}
