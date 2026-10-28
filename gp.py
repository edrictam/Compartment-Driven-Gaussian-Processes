"""Covariance kernels, GP inference, and exposure functional posteriors.

The noise model is generic. Throughout this module, the observation-noise
specification `sigma_eps` may be either
  - a scalar: homoscedastic noise variance sigma_eps**2 applied to all
    observations, or
  - a 1-D array of length n = len(T): heteroscedastic per-observation noise
    standard deviations, applied as diag(sigma_eps**2).
The former case covers individualized patient data with a common assay
variance; the latter covers aggregated cohort-mean data with reported
per-timepoint standard errors.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize

# Stationary covariance kernels

def matern_52(s, t, sigma_f, ell):
    """Matern-5/2: sigma_f^2 (1 + sqrt(5) u + 5 u^2 / 3) exp(-sqrt(5) u)."""
    u = np.abs(np.asarray(s) - np.asarray(t)) / ell
    return sigma_f ** 2 * (1.0 + np.sqrt(5.0) * u + 5.0 * u ** 2 / 3.0) * np.exp(-np.sqrt(5.0) * u)


def matern_32(s, t, sigma_f, ell):
    """Matern-3/2: sigma_f^2 (1 + sqrt(3) u) exp(-sqrt(3) u)."""
    u = np.abs(np.asarray(s) - np.asarray(t)) / ell
    return sigma_f ** 2 * (1.0 + np.sqrt(3.0) * u) * np.exp(-np.sqrt(3.0) * u)


def matern_12(s, t, sigma_f, ell):
    """Matern-1/2 (Ornstein-Uhlenbeck): sigma_f^2 exp(-u)."""
    u = np.abs(np.asarray(s) - np.asarray(t)) / ell
    return sigma_f ** 2 * np.exp(-u)


def squared_exponential(s, t, sigma_f, ell):
    """Squared exponential / RBF: sigma_f^2 exp(-u^2 / 2)."""
    u = np.abs(np.asarray(s) - np.asarray(t)) / ell
    return sigma_f ** 2 * np.exp(-0.5 * u ** 2)


KERNELS = {
    "matern52": matern_52,
    "matern32": matern_32,
    "matern12": matern_12,
    "se":       squared_exponential,
}


def kernel_matrix(T, U, sigma_f, ell, kernel=matern_12):
    """Build the n x q matrix of kernel evaluations between T and U.
    `kernel` may be a callable or a string key into KERNELS."""
    if isinstance(kernel, str):
        kernel = KERNELS[kernel]
    T = np.atleast_1d(np.asarray(T))
    U = np.atleast_1d(np.asarray(U))
    return kernel(T[:, None], U[None, :], sigma_f, ell)


def _noise_var_diag(sigma_eps, n):
    """Return the length-n diagonal variance vector for the noise model.
    Accepts scalar or length-n array `sigma_eps`."""
    arr = np.atleast_1d(np.asarray(sigma_eps, dtype=float))
    if arr.size == 1:
        return np.full(n, float(arr[0]) ** 2)
    if arr.size != n:
        raise ValueError(
            f"sigma_eps array length {arr.size} does not match n_obs {n}")
    return arr ** 2


# ---------------------------------------------------------------------------
# Posterior latent curve
# ---------------------------------------------------------------------------

def gp_posterior(T, y, U, mean_T, mean_U, sigma_f, ell, sigma_eps,
                 jitter=1e-8, kernel="matern12"):
    """Posterior GP at the prediction grid U given training observations.

    Returns the posterior mean vector mu(U) of length q and the q x q
    posterior covariance K^+(U, U). `kernel` is a key into KERNELS or a
    callable kernel function. `sigma_eps` may be a scalar (homoscedastic
    noise) or a length-n vector (heteroscedastic noise).
    """
    n = len(T)
    K_TT = kernel_matrix(T, T, sigma_f, ell, kernel=kernel)
    noise_var = _noise_var_diag(sigma_eps, n)
    V = K_TT + np.diag(noise_var) + jitter * np.eye(n)
    L_V, low = cho_factor(V, lower=True)

    resid = y - mean_T
    alpha = cho_solve((L_V, low), resid)

    K_UT = kernel_matrix(U, T, sigma_f, ell, kernel=kernel)
    K_UU = kernel_matrix(U, U, sigma_f, ell, kernel=kernel)

    mu_U = mean_U + K_UT @ alpha
    Vinv_K_TU = cho_solve((L_V, low), K_UT.T)
    K_plus = K_UU - K_UT @ Vinv_K_TU
    K_plus = 0.5 * (K_plus + K_plus.T)
    return mu_U, K_plus


def neg_log_marginal_likelihood(params_log, T, y, mean_T, jitter=1e-8,
                                 kernel="matern12",
                                 log_prior_mean=None, log_prior_sd=None):
    """Negative log marginal likelihood; estimates sigma_f, ell, sigma_eps
    (homoscedastic scalar noise model).

    If `log_prior_mean` and `log_prior_sd` are given (length-3 arrays ordered
    (log sigma_f, log ell, log sigma_eps)), a Gaussian penalty on the
    log-hyperparameters is added, turning the objective into a MAP
    (penalized marginal likelihood). A coordinate with log_prior_sd = inf is
    left unpenalized (flat over the numerical box)."""
    sigma_f, ell, sigma_eps = np.exp(params_log)
    n = len(T)
    K_TT = kernel_matrix(T, T, sigma_f, ell, kernel=kernel)
    V = K_TT + (sigma_eps ** 2) * np.eye(n) + jitter * np.eye(n)
    try:
        L_V, low = cho_factor(V, lower=True)
    except np.linalg.LinAlgError:
        return 1e10
    resid = y - mean_T
    alpha = cho_solve((L_V, low), resid)
    log_det = 2.0 * np.sum(np.log(np.diag(L_V)))
    nll = 0.5 * (resid @ alpha + log_det + n * np.log(2.0 * np.pi))
    if log_prior_mean is not None:
        z = (params_log - log_prior_mean) / log_prior_sd
        nll = nll + 0.5 * np.dot(z, z)
    return nll


def neg_log_marginal_likelihood_fixed_noise(params_log_2, T, y, mean_T,
                                             noise_var_diag,
                                             jitter=1e-8,
                                             kernel="matern12",
                                             log_prior_mean=None,
                                             log_prior_sd=None):
    """Negative log marginal likelihood with fixed (known) heteroscedastic
    noise variances. Estimates only sigma_f, ell.

    If `log_prior_mean`/`log_prior_sd` are given (length-2 arrays ordered
    (log sigma_f, log ell)), a Gaussian penalty on the log-hyperparameters is
    added (MAP); a coordinate with log_prior_sd = inf is left unpenalized."""
    sigma_f, ell = np.exp(params_log_2)
    n = len(T)
    K_TT = kernel_matrix(T, T, sigma_f, ell, kernel=kernel)
    V = K_TT + np.diag(noise_var_diag) + jitter * np.eye(n)
    try:
        L_V, low = cho_factor(V, lower=True)
    except np.linalg.LinAlgError:
        return 1e10
    resid = y - mean_T
    alpha = cho_solve((L_V, low), resid)
    log_det = 2.0 * np.sum(np.log(np.diag(L_V)))
    nll = 0.5 * (resid @ alpha + log_det + n * np.log(2.0 * np.pi))
    if log_prior_mean is not None:
        z = (params_log_2 - log_prior_mean) / log_prior_sd
        nll = nll + 0.5 * np.dot(z, z)
    return nll


def fit_hyperparameters_fixed_noise(T, y, mean_T, sigma_eps_fixed,
                                     n_restarts=2, rng=None,
                                     sigma_f_init=1.0, ell_init=4.0,
                                     bounds_log_2=((-6, 4), (-4, 4)),
                                     kernel="matern12",
                                     sigma_f_scale=None, ell_scale=None,
                                     prior_log_sd=1.0):
    """MAP estimate of (sigma_f, ell) with fixed (known) per-observation
    noise. `sigma_eps_fixed` is a scalar or length-n vector treated as known.
    Returns (sigma_f, ell, sigma_eps_fixed).

    `sigma_f_scale` and `ell_scale` are the medians of independent log-normal
    priors on sigma_f and ell (equivalently Gaussian priors on the log
    parameters with log-SD `prior_log_sd`); a `None` scale leaves that
    coordinate unpenalized (flat over the numerical box). The two-sided
    log-normal on `ell` penalizes both ell -> 0 and ell -> infinity, replacing
    the old hard ell ceiling that prevented the GP discrepancy from
    degenerating into a global level shift confounded with the mechanistic
    mean. The wide `bounds_log_2` box is retained purely for numerical
    conditioning."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(T)
    noise_var = _noise_var_diag(sigma_eps_fixed, n)

    lp_mean = np.array([np.log(sigma_f_scale) if sigma_f_scale else 0.0,
                        np.log(ell_scale)      if ell_scale     else 0.0])
    lp_sd = np.array([prior_log_sd if sigma_f_scale else np.inf,
                      prior_log_sd if ell_scale     else np.inf])

    inits = [(sigma_f_scale or sigma_f_init,
              ell_scale or ell_init)]
    for _ in range(n_restarts - 1):
        inits.append((np.exp(rng.uniform(-1.5, 1.5)),
                      np.exp(rng.uniform(0.0, 2.5))))

    best = None
    best_val = np.inf
    for sf, el in inits:
        x0 = np.log([sf, el])
        try:
            res = minimize(
                neg_log_marginal_likelihood_fixed_noise, x0,
                args=(T, y, mean_T, noise_var, 1e-8, kernel, lp_mean, lp_sd),
                method="L-BFGS-B",
                bounds=bounds_log_2,
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x
        except Exception:
            continue
    if best is None:
        return sigma_f_init, ell_init, sigma_eps_fixed
    sigma_f, ell = np.exp(best)
    return sigma_f, ell, sigma_eps_fixed


def fit_hyperparameters_fixed_ell(T, y, mean_T, ell_fixed, n_restarts=2, rng=None,
                                  sigma_f_init=1.0, sigma_eps_init=0.1,
                                  bounds_log_2=((-6, 4), (-6, 2)),
                                  kernel="matern12",
                                  sigma_f_scale=None, sigma_eps_scale=None,
                                  prior_log_sd=1.0):
    """MAP estimate of (sigma_f, sigma_eps) with the length scale held fixed.

    Fixing `ell` rather than penalizing it removes the length-scale coordinate
    from the optimization entirely, so no numerical box bound can be attained
    in that coordinate. Returns (sigma_f, ell_fixed, sigma_eps).
    """
    from scipy.optimize import minimize
    if rng is None:
        rng = np.random.default_rng(0)
    lp_mean = np.array([np.log(sigma_f_scale)  if sigma_f_scale  else 0.0,
                        np.log(sigma_eps_scale) if sigma_eps_scale else 0.0])
    lp_sd = np.array([prior_log_sd if sigma_f_scale  else np.inf,
                      prior_log_sd if sigma_eps_scale else np.inf])

    def obj(z):
        params = np.array([z[0], np.log(ell_fixed), z[1]])
        nll = neg_log_marginal_likelihood(params, T, y, mean_T, 1e-8, kernel,
                                          np.array([lp_mean[0], 0.0, lp_mean[1]]),
                                          np.array([lp_sd[0], np.inf, lp_sd[1]]))
        return nll

    best, best_val = None, np.inf
    inits = [np.log([sigma_f_init, sigma_eps_init])]
    for _ in range(max(0, n_restarts - 1)):
        inits.append(np.log([sigma_f_init, sigma_eps_init]) + rng.normal(0, 0.5, size=2))
    for x0 in inits:
        x0 = np.clip(x0, [b[0] for b in bounds_log_2], [b[1] for b in bounds_log_2])
        try:
            r = minimize(obj, x0, method="L-BFGS-B", bounds=bounds_log_2)
        except Exception:
            continue
        if r.success or np.isfinite(r.fun):
            if r.fun < best_val:
                best_val, best = r.fun, r.x
    if best is None:
        return sigma_f_init, float(ell_fixed), sigma_eps_init
    sf, se = np.exp(best)
    return float(sf), float(ell_fixed), float(se)


def fit_hyperparameters_fixed_ell_noise(T, y, mean_T, ell_fixed,
                                        sigma_eps_fixed, n_restarts=2, rng=None,
                                        sigma_f_init=1.0, bounds_log=(-6, 4),
                                        kernel="matern12", sigma_f_scale=None,
                                        prior_log_sd=1.0):
    """Fit only the residual SD when both length scale and noise are known.

    In particular, retain the supplied per-observation SE vector for cohort
    means rather than replacing it by a fitted homoscedastic noise level.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if ell_fixed <= 0:
        raise ValueError("ell_fixed must be positive")
    noise_var = _noise_var_diag(sigma_eps_fixed, len(T))
    lp_mean = np.array([np.log(sigma_f_scale) if sigma_f_scale else 0.0, 0.0])
    lp_sd = np.array([prior_log_sd if sigma_f_scale else np.inf, np.inf])

    def objective(z):
        return neg_log_marginal_likelihood_fixed_noise(
            np.array([z[0], np.log(ell_fixed)]), T, y, mean_T, noise_var,
            kernel=kernel, log_prior_mean=lp_mean, log_prior_sd=lp_sd)

    initial = np.log(sigma_f_scale or sigma_f_init)
    starts = [initial] + [initial + rng.normal(0, 0.5)
                          for _ in range(max(0, n_restarts - 1))]
    candidates = []
    for value in starts:
        result = minimize(objective, [np.clip(value, *bounds_log)],
                          method="L-BFGS-B", bounds=[bounds_log])
        if np.isfinite(result.fun) and result.success:
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Fixed-length-scale, fixed-noise MAP optimization failed")
    best = min(candidates, key=lambda result: result.fun)
    return float(np.exp(best.x[0])), float(ell_fixed), sigma_eps_fixed


def fit_hyperparameters(T, y, mean_T, n_restarts=2, rng=None,
                        sigma_f_init=1.0, ell_init=4.0, sigma_eps_init=0.1,
                        bounds_log=((-6, 4), (-4, 4), (-6, 2)),
                        kernel="matern12",
                        sigma_f_scale=None, sigma_eps_scale=None,
                        ell_scale=None, prior_log_sd=1.0):
    """MAP estimate of kernel + noise hyperparameters via L-BFGS-B with
    multistart. Returns the best (sigma_f, ell, sigma_eps).

    `sigma_f_scale`, `sigma_eps_scale`, `ell_scale` are the medians of
    independent log-normal priors on the respective hyperparameters
    (equivalently Gaussian priors on the log parameters with log-SD
    `prior_log_sd`); a `None` scale leaves that coordinate unpenalized (flat
    over the numerical box). The log-normal priors on sigma_f and sigma_eps
    penalize the collapse-to-zero failure mode of type-II maximum likelihood
    in small samples, replacing the old hard variance floors. The wide
    `bounds_log` box is retained purely for numerical conditioning.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    lp_mean = np.array([np.log(sigma_f_scale)   if sigma_f_scale   else 0.0,
                        np.log(ell_scale)        if ell_scale       else 0.0,
                        np.log(sigma_eps_scale)  if sigma_eps_scale else 0.0])
    lp_sd = np.array([prior_log_sd if sigma_f_scale   else np.inf,
                      prior_log_sd if ell_scale       else np.inf,
                      prior_log_sd if sigma_eps_scale else np.inf])
    inits = [(sigma_f_scale or sigma_f_init,
              ell_scale or ell_init,
              sigma_eps_scale or sigma_eps_init)]
    for _ in range(n_restarts - 1):
        inits.append((np.exp(rng.uniform(-1.5, 1.5)),
                      np.exp(rng.uniform(0.0, 2.5)),
                      np.exp(rng.uniform(-2.5, -0.5))))
    best = None
    best_val = np.inf
    for sf, el, se in inits:
        x0 = np.log([sf, el, se])
        try:
            res = minimize(
                neg_log_marginal_likelihood, x0,
                args=(T, y, mean_T, 1e-8, kernel, lp_mean, lp_sd),
                method="L-BFGS-B",
                bounds=bounds_log,
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x
        except Exception:
            continue
    if best is None:
        return sigma_f_init, ell_init, sigma_eps_init
    sigma_f, ell, sigma_eps = np.exp(best)
    return sigma_f, ell, sigma_eps


# ---------------------------------------------------------------------------
# Functional posteriors
# ---------------------------------------------------------------------------

from functools import lru_cache


@lru_cache(maxsize=64)
def _leggauss_cached(q):
    return np.polynomial.legendre.leggauss(q)


def gauss_legendre_nodes(a, b, q):
    nodes, weights = _leggauss_cached(int(q))
    nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    weights = 0.5 * (b - a) * weights
    return nodes, weights


def auc_posterior_gaussian(mu_U, K_UU, weights):
    """Closed-form Gaussian posterior of weighted AUC under the GP posterior.

    Returns (mean, variance) of omega^T C(U).
    """
    mean = float(weights @ mu_U)
    var = float(weights @ K_UU @ weights)
    return mean, max(var, 0.0)


def sample_posterior_paths(mu_U, K_UU, n_paths, rng, jitter=1e-8):
    """Sample n_paths posterior paths on U via Cholesky factorization of
    K_UU + jitter I."""
    q = len(mu_U)
    trace = np.trace(K_UU)
    eta = max(jitter, 1e-8 * trace / q)
    K = K_UU + eta * np.eye(q)
    # Adaptive jitter inflation if Cholesky fails.
    for k in range(6):
        try:
            L = cholesky(K, lower=True)
            break
        except np.linalg.LinAlgError:
            eta *= 10.0
            K = K_UU + eta * np.eye(q)
    else:
        # Fall back to eigen-decomposition.
        evals, evecs = np.linalg.eigh(K_UU)
        evals = np.clip(evals, 0.0, None)
        L = evecs @ np.diag(np.sqrt(evals))
    Z = rng.standard_normal(size=(q, n_paths))
    return mu_U[:, None] + L @ Z  # shape (q, n_paths)


def cmax_posterior_samples(paths):
    """paths has shape (q, n_paths). Returns vector of Cmax samples."""
    return paths.max(axis=0)


def tmax_posterior_samples(paths, grid):
    """paths has shape (q, n_paths). Returns vector of Tmax samples (in grid units)."""
    idx = paths.argmax(axis=0)
    return grid[idx]
