"""Patient-level estimation methods compared in the simulation study.

Each method takes the patient's observed sampling times and concentrations,
along with model-specific information (dose, route, time window), and returns
point estimates and 95% credible / confidence intervals for AUC and Cmax.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import models, gp


# ---------------------------------------------------------------------------
# Noncompartmental analysis
# ---------------------------------------------------------------------------

def trapezoidal_auc(t_obs, y_obs):
    order = np.argsort(t_obs)
    t = np.asarray(t_obs)[order]
    y = np.clip(np.asarray(y_obs)[order], 0.0, None)
    return float(np.trapezoid(y, t))


def method_nca(t_obs, y_obs):
    return {
        "auc": trapezoidal_auc(t_obs, y_obs),
        "auc_lo": np.nan,
        "auc_hi": np.nan,
        "cmax": float(max(y_obs)),
        "cmax_lo": np.nan,
        "cmax_hi": np.nan,
    }


def method_nca_bootstrap(t_obs, y_obs, n_boot=500, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(t_obs)
    aucs = np.empty(n_boot)
    cmaxs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        idx = np.unique(idx)
        if len(idx) < 2:
            idx = np.arange(n)
        aucs[b] = trapezoidal_auc(np.asarray(t_obs)[idx], np.asarray(y_obs)[idx])
        cmaxs[b] = float(max(np.asarray(y_obs)[idx]))
    auc_lo, auc_hi = np.quantile(aucs, [0.025, 0.975])
    cmax_lo, cmax_hi = np.quantile(cmaxs, [0.025, 0.975])
    return {
        "auc": trapezoidal_auc(t_obs, y_obs),
        "auc_lo": float(auc_lo),
        "auc_hi": float(auc_hi),
        "cmax": float(max(y_obs)),
        "cmax_lo": float(cmax_lo),
        "cmax_hi": float(cmax_hi),
    }


# ---------------------------------------------------------------------------
# Parametric compartmental fits with parametric bootstrap intervals
# ---------------------------------------------------------------------------

def _fit_1cpt_oral_nls(t_obs, y_obs, dose, x0=None):
    def residuals(theta_log):
        ka, ke, V = np.exp(theta_log)
        return models.one_cpt_oral(t_obs, ka, ke, V, dose) - y_obs
    if x0 is None:
        x0 = np.log([1.5, 0.1, 30.0])
    try:
        res = least_squares(residuals, x0, method="lm")
        ka, ke, V = np.exp(res.x)
        return ka, ke, V, np.sqrt(np.mean(res.fun ** 2) + 1e-12)
    except Exception:
        return 1.5, 0.1, 30.0, 0.1


def _fit_1cpt_iv_nls(t_obs, y_obs, dose, x0=None):
    def residuals(theta_log):
        ke, V = np.exp(theta_log)
        return models.one_cpt_iv(t_obs, ke, V, dose) - y_obs
    if x0 is None:
        x0 = np.log([0.1, 30.0])
    try:
        res = least_squares(residuals, x0, method="lm")
        ke, V = np.exp(res.x)
        return ke, V, np.sqrt(np.mean(res.fun ** 2) + 1e-12)
    except Exception:
        return 0.1, 30.0, 0.1


def _fit_2cpt_oral_nls(t_obs, y_obs, dose, x0=None):
    def residuals(theta_log):
        ka, ke, k12, k21, V = np.exp(theta_log)
        try:
            pred = models.two_cpt_oral(t_obs, ka, ke, k12, k21, V, dose)
        except Exception:
            return 1e6 * np.ones_like(y_obs)
        return pred - y_obs
    if x0 is None:
        x0 = np.log([1.5, 0.1, 0.2, 0.1, 30.0])
    try:
        res = least_squares(residuals, x0, method="lm", max_nfev=200)
        ka, ke, k12, k21, V = np.exp(res.x)
        return ka, ke, k12, k21, V, np.sqrt(np.mean(res.fun ** 2) + 1e-12)
    except Exception:
        return 1.5, 0.1, 0.2, 0.1, 30.0, 0.1


def _auc_cmax_from_model_grid(t_grid, weights_grid, curve):
    auc = float(weights_grid @ curve)
    cmax = float(curve.max())
    return auc, cmax


def method_parametric_1cpt_oral(t_obs, y_obs, dose, a, b, rng=None, n_boot=25):
    """Parametric 1-cpt oral fit with a light parametric bootstrap."""
    if rng is None:
        rng = np.random.default_rng(0)
    ka, ke, V, sigma_hat = _fit_1cpt_oral_nls(t_obs, y_obs, dose)
    t_grid, w_grid = gp.gauss_legendre_nodes(a, b, 64)
    curve_hat = models.one_cpt_oral(t_grid, ka, ke, V, dose)
    auc_hat, cmax_hat = _auc_cmax_from_model_grid(t_grid, w_grid, curve_hat)

    fit_pred = models.one_cpt_oral(t_obs, ka, ke, V, dose)
    aucs = np.empty(n_boot); cmaxs = np.empty(n_boot)
    x0 = np.log([ka, ke, V])
    for j in range(n_boot):
        y_sim = fit_pred + rng.normal(0.0, sigma_hat, size=len(t_obs))
        ka_b, ke_b, V_b, _ = _fit_1cpt_oral_nls(t_obs, y_sim, dose, x0=x0)
        curve_b = models.one_cpt_oral(t_grid, ka_b, ke_b, V_b, dose)
        aucs[j], cmaxs[j] = _auc_cmax_from_model_grid(t_grid, w_grid, curve_b)
    return {
        "auc": auc_hat,
        "auc_lo": float(np.quantile(aucs, 0.025)),
        "auc_hi": float(np.quantile(aucs, 0.975)),
        "cmax": cmax_hat,
        "cmax_lo": float(np.quantile(cmaxs, 0.025)),
        "cmax_hi": float(np.quantile(cmaxs, 0.975)),
    }


def method_parametric_1cpt_iv(t_obs, y_obs, dose, a, b, rng=None, n_boot=25):
    """Parametric 1-cpt IV bolus fit with a light parametric bootstrap."""
    if rng is None:
        rng = np.random.default_rng(0)
    ke, V, sigma_hat = _fit_1cpt_iv_nls(t_obs, y_obs, dose)
    t_grid, w_grid = gp.gauss_legendre_nodes(a, b, 64)
    curve_hat = models.one_cpt_iv(t_grid, ke, V, dose)
    auc_hat, cmax_hat = _auc_cmax_from_model_grid(t_grid, w_grid, curve_hat)

    fit_pred = models.one_cpt_iv(t_obs, ke, V, dose)
    aucs = np.empty(n_boot); cmaxs = np.empty(n_boot)
    x0 = np.log([ke, V])
    for j in range(n_boot):
        y_sim = fit_pred + rng.normal(0.0, sigma_hat, size=len(t_obs))
        ke_b, V_b, _ = _fit_1cpt_iv_nls(t_obs, y_sim, dose, x0=x0)
        curve_b = models.one_cpt_iv(t_grid, ke_b, V_b, dose)
        aucs[j], cmaxs[j] = _auc_cmax_from_model_grid(t_grid, w_grid, curve_b)
    return {
        "auc": auc_hat,
        "auc_lo": float(np.quantile(aucs, 0.025)),
        "auc_hi": float(np.quantile(aucs, 0.975)),
        "cmax": cmax_hat,
        "cmax_lo": float(np.quantile(cmaxs, 0.025)),
        "cmax_hi": float(np.quantile(cmaxs, 0.975)),
    }


def method_parametric_2cpt_oral(t_obs, y_obs, dose, a, b, rng=None, n_boot=25):
    """Parametric 2-cpt oral fit with a light parametric bootstrap."""
    if rng is None:
        rng = np.random.default_rng(0)
    ka, ke, k12, k21, V, sigma_hat = _fit_2cpt_oral_nls(t_obs, y_obs, dose)
    t_grid, w_grid = gp.gauss_legendre_nodes(a, b, 64)
    try:
        curve_hat = models.two_cpt_oral(t_grid, ka, ke, k12, k21, V, dose)
    except Exception:
        curve_hat = models.one_cpt_oral(t_grid, ka, ke, V, dose)
    auc_hat, cmax_hat = _auc_cmax_from_model_grid(t_grid, w_grid, curve_hat)

    try:
        fit_pred = models.two_cpt_oral(t_obs, ka, ke, k12, k21, V, dose)
    except Exception:
        fit_pred = models.one_cpt_oral(t_obs, ka, ke, V, dose)
    aucs = []; cmaxs = []
    x0 = np.log([ka, ke, k12, k21, V])
    for j in range(n_boot):
        y_sim = fit_pred + rng.normal(0.0, sigma_hat, size=len(t_obs))
        ka_b, ke_b, k12_b, k21_b, V_b, _ = _fit_2cpt_oral_nls(
            t_obs, y_sim, dose, x0=x0)
        try:
            curve_b = models.two_cpt_oral(t_grid, ka_b, ke_b, k12_b, k21_b, V_b, dose)
            a_b, c_b = _auc_cmax_from_model_grid(t_grid, w_grid, curve_b)
            aucs.append(a_b); cmaxs.append(c_b)
        except Exception:
            continue
    if len(aucs) < 5:
        return {"auc": auc_hat, "auc_lo": np.nan, "auc_hi": np.nan,
                "cmax": cmax_hat, "cmax_lo": np.nan, "cmax_hi": np.nan}
    aucs = np.array(aucs); cmaxs = np.array(cmaxs)
    return {
        "auc": auc_hat,
        "auc_lo": float(np.quantile(aucs, 0.025)),
        "auc_hi": float(np.quantile(aucs, 0.975)),
        "cmax": cmax_hat,
        "cmax_lo": float(np.quantile(cmaxs, 0.025)),
        "cmax_hi": float(np.quantile(cmaxs, 0.975)),
    }


# ---------------------------------------------------------------------------
# Gaussian process methods (pure GP and mechanistically centered GP)
# ---------------------------------------------------------------------------

def _gp_method(t_obs, y_obs, mean_func, a, b, n_paths=2000, rng=None,
               n_grid_path=200, n_quad=64, kernel="matern12",
               sigma_f_scale=None, sigma_eps_scale=None, ell_scale=None,
               sigma_eps_fixed=None, ell_fixed=None, prior_log_sd=1.0):
    """Generic GP posterior method.

    If `sigma_eps_fixed` is given (scalar or length-n vector), the noise is
    treated as known and only (sigma_f, ell) are estimated by MAP. This is
    the aggregated cohort-mean setting where the per-timepoint standard error
    is reported alongside the cohort-mean concentration. The `*_scale`
    arguments are the medians of the weakly-informative log-normal
    hyperparameter priors (see `gp.fit_hyperparameters`).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    t_obs = np.asarray(t_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    mean_T = mean_func(t_obs)

    if ell_fixed is not None and sigma_eps_fixed is not None:
        sigma_f, ell, sigma_eps = gp.fit_hyperparameters_fixed_ell_noise(
            t_obs, y_obs, mean_T, ell_fixed, sigma_eps_fixed, rng=rng,
            kernel=kernel, sigma_f_scale=sigma_f_scale,
            prior_log_sd=prior_log_sd)
    elif ell_fixed is not None:
        sigma_f, ell, sigma_eps = gp.fit_hyperparameters_fixed_ell(
            t_obs, y_obs, mean_T, ell_fixed, rng=rng, kernel=kernel,
            sigma_f_scale=sigma_f_scale, sigma_eps_scale=sigma_eps_scale,
            prior_log_sd=prior_log_sd)
    elif sigma_eps_fixed is None:
        sigma_f, ell, sigma_eps = gp.fit_hyperparameters(
            t_obs, y_obs, mean_T, rng=rng, kernel=kernel,
            sigma_f_scale=sigma_f_scale, sigma_eps_scale=sigma_eps_scale,
            ell_scale=ell_scale, prior_log_sd=prior_log_sd)
    else:
        sigma_f, ell, sigma_eps = gp.fit_hyperparameters_fixed_noise(
            t_obs, y_obs, mean_T, sigma_eps_fixed, rng=rng, kernel=kernel,
            sigma_f_scale=sigma_f_scale, ell_scale=ell_scale,
            prior_log_sd=prior_log_sd)

    t_quad, w_quad = gp.gauss_legendre_nodes(a, b, n_quad)
    mu_quad, K_quad = gp.gp_posterior(t_obs, y_obs, t_quad, mean_T,
                                       mean_func(t_quad),
                                       sigma_f, ell, sigma_eps, kernel=kernel)
    auc_mean, auc_var = gp.auc_posterior_gaussian(mu_quad, K_quad, w_quad)
    auc_sd = np.sqrt(auc_var)

    t_grid = np.linspace(a, b, n_grid_path)
    mu_grid, K_grid = gp.gp_posterior(t_obs, y_obs, t_grid, mean_T,
                                       mean_func(t_grid),
                                       sigma_f, ell, sigma_eps, kernel=kernel)
    paths = gp.sample_posterior_paths(mu_grid, K_grid, n_paths, rng)
    cmax_samples = gp.cmax_posterior_samples(paths)
    cmax_lo, cmax_hi = np.quantile(cmax_samples, [0.025, 0.975])

    # Draw posterior AUC samples too, useful for downstream functionals
    # such as clearance (CL = dose / AUC) and for comparing AUC posteriors
    # across cohorts.
    auc_samples = rng.normal(auc_mean, auc_sd, size=n_paths)

    return {
        "auc": auc_mean,
        "auc_sd": float(auc_sd),
        "auc_lo": float(auc_mean - 1.96 * auc_sd),
        "auc_hi": float(auc_mean + 1.96 * auc_sd),
        "auc_samples": auc_samples,
        "cmax": float(cmax_samples.mean()),
        "cmax_lo": float(cmax_lo),
        "cmax_hi": float(cmax_hi),
        "cmax_samples": cmax_samples,
        "_paths_grid": t_grid,
        "_paths": paths,
        "_mu_grid": mu_grid,
        "_K_grid_diag": np.diag(K_grid),
        "_hyper": (sigma_f, ell, sigma_eps),
    }


def method_pure_gp(t_obs, y_obs, a, b, n_paths=2000, rng=None,
                   kernel="matern12", n_grid_path=200,
                   sigma_f_scale=None, sigma_eps_scale=None, ell_scale=None,
                   ell_fixed=None, prior_log_sd=1.0):
    """Pure GP with a zero mean function.

    This is the no-mechanistic-information baseline: the prior mean carries
    no pharmacokinetic structure, so the Gaussian process alone must account
    for the entire concentration--time curve rather than for a residual
    around a compartmental mean.

    Because of that, the marginal-SD prior for this method has to be pinned
    to the scale of the curve itself, not to the scale of a discrepancy. If
    the caller does not supply `sigma_f_scale`, we set its prior median to
    the root-mean-square of the observations, which is the same
    "pin the prior to the concentration scale of the data" rule used for the
    compartment-driven prior. Passing the compartment-driven residual scale
    here instead would understate the required marginal variance by roughly
    an order of magnitude and handicap the baseline unfairly.
    """
    y = np.asarray(y_obs, dtype=float)
    def mean_func(t):
        return np.zeros_like(np.asarray(t, dtype=float))
    if sigma_f_scale is None:
        rms = float(np.sqrt(np.mean(y ** 2)))
        sigma_f_scale = rms if rms > 0 else None
    return _gp_method(t_obs, y_obs, mean_func, a, b, n_paths=n_paths, rng=rng,
                      kernel=kernel, n_grid_path=n_grid_path,
                      sigma_f_scale=sigma_f_scale, sigma_eps_scale=sigma_eps_scale,
                      ell_scale=ell_scale, ell_fixed=ell_fixed,
                      prior_log_sd=prior_log_sd)


def method_mech_gp(t_obs, y_obs, mean_func, a, b, n_paths=2000, rng=None,
                   kernel="matern12", n_grid_path=200,
                   sigma_f_scale=None, sigma_eps_scale=None, ell_scale=None,
                   sigma_eps_fixed=None, ell_fixed=None, prior_log_sd=1.0):
    """Compartment-driven GP: caller supplies the compartmental mean function
    already specialized to the chosen drug/route and pop parameters.

    When `sigma_eps_fixed` is None (default), observation noise is fit by MAP
    as a scalar (individualized data setting). When `sigma_eps_fixed` is a
    scalar or length-n vector, it is treated as known per-observation noise
    (aggregated cohort-mean data setting with reported per-timepoint standard
    errors)."""
    return _gp_method(t_obs, y_obs, mean_func, a, b, n_paths=n_paths, rng=rng,
                      kernel=kernel, n_grid_path=n_grid_path,
                      sigma_f_scale=sigma_f_scale, sigma_eps_scale=sigma_eps_scale,
                      ell_scale=ell_scale, sigma_eps_fixed=sigma_eps_fixed,
                      ell_fixed=ell_fixed,
                      prior_log_sd=prior_log_sd)
