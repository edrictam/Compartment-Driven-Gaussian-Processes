"""Scenario specifications and the per-scenario simulation driver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import models, methods, gp


# Standard and sparse sampling grids (hours post-dose). Both designs place
# samples at 1 and 2 hr so that the absorption peak (t_max ~ 1.9 hr for the
# population-typical oral parameters) is bracketed; estimating a supremum
# from a design that never samples near the peak is a limitation of the
# design rather than of the estimator.
STANDARD_GRID = np.array([0.5, 1.0, 1.5, 2.0, 3.0,
                          4.0, 6.0, 8.0, 12.0, 24.0])   # 10 samples
SPARSE_GRID   = np.array([1.0, 2.0, 4.0, 8.0, 16.0])    # 5 samples
TIME_WINDOW = (0.0, 24.0)
DOSE = 300.0
SIGMA_EPS = 0.1


@dataclass
class Scenario:
    name: str
    generating_model: str   # 'one_cpt_oral' | 'two_cpt_oral' | 'one_cpt_iv'
    fitted_mean: str        # 'one_cpt_oral' | 'one_cpt_iv'
    sampling: str           # 'standard' | 'sparse'
    description: str = ""
    # Non-compartmental discrepancy added to the true curve. tau = 0 recovers a
    # purely compartmental truth; tau > 0 draws delta ~ GP(0, tau^2 Matern-5/2)
    # per patient, so the truth no longer lies on the 3-parameter manifold.
    tau: float = 0.0
    ell_delta: float = 4.0

    @property
    def grid(self):
        return STANDARD_GRID if self.sampling == "standard" else SPARSE_GRID


SCENARIOS = {
    "S1": Scenario("S1", "one_cpt_oral", "one_cpt_oral", "standard",
                   "Correctly specified one-compartment oral."),
    "S2": Scenario("S2", "two_cpt_oral", "one_cpt_oral", "standard",
                   "Two-compartment generator, one-compartment oral prior mean (misspecified)."),
    "S3": Scenario("S3", "one_cpt_iv",  "one_cpt_iv",   "standard",
                   "Correctly specified one-compartment IV bolus."),
    "S4": Scenario("S4", "one_cpt_oral", "one_cpt_oral", "sparse",
                   "Correctly specified one-compartment oral, sparse sampling."),
    "S5": Scenario("S5", "two_cpt_oral", "one_cpt_oral", "sparse",
                   "Two-compartment generator with one-compartment oral prior mean, sparse sampling."),
    "S6": Scenario("S6", "one_cpt_iv",  "one_cpt_iv",   "sparse",
                   "Correctly specified one-compartment IV bolus, sparse sampling."),
    # S7-S9 add a non-compartmental discrepancy to the true curve, so the truth
    # leaves the 3-parameter compartmental manifold. tau matches the method's
    # own sigma_f prior median, i.e. these are the correctly-specified-prior
    # scenarios.
    "S7": Scenario("S7", "one_cpt_oral", "one_cpt_oral", "standard",
                   "One-compartment oral with GP discrepancy.", tau=0.6),
    "S8": Scenario("S8", "two_cpt_oral", "one_cpt_oral", "standard",
                   "Two-compartment generator with GP discrepancy, one-compartment oral prior mean.",
                   tau=0.6),
    "S9": Scenario("S9", "one_cpt_oral", "one_cpt_oral", "sparse",
                   "One-compartment oral with GP discrepancy, sparse sampling.", tau=0.6),
}

# Deterministic per-scenario seed offsets. Python randomizes string hashing
# between interpreter runs unless PYTHONHASHSEED is fixed, so hash(name) would
# make the reported simulation results irreproducible.
SEED_OFFSET = {f"S{i}": i for i in range(1, 10)}


# Population parameters used as ground truth in the simulation and as the
# mechanistically informed prior mean.
POP_PARAMS = {
    "ka_pop": 1.5, "ke_pop": 0.1, "c0_pop": 10.0,
    "k12_pop": 0.2, "k21_pop": 0.1,
    "omega_ka": 0.3, "omega_ke": 0.02, "omega_c0": 2.0,
    "omega_k12": 0.04, "omega_k21": 0.02,
}


# ---------------------------------------------------------------------------
# Truth generators and prior-mean functions
# ---------------------------------------------------------------------------

def draw_true_params(scenario: Scenario, rng, N):
    if scenario.generating_model == "one_cpt_oral":
        ka, ke, V = models.draw_one_cpt_oral_params(
            rng, N, POP_PARAMS["ka_pop"], POP_PARAMS["ke_pop"], POP_PARAMS["c0_pop"],
            POP_PARAMS["omega_ka"], POP_PARAMS["omega_ke"], POP_PARAMS["omega_c0"],
            dose=DOSE)
        return [{"ka": ka[i], "ke": ke[i], "V": V[i]} for i in range(N)]
    if scenario.generating_model == "two_cpt_oral":
        ka, ke, V, k12, k21 = models.draw_two_cpt_oral_params(
            rng, N, POP_PARAMS["ka_pop"], POP_PARAMS["ke_pop"], POP_PARAMS["c0_pop"],
            POP_PARAMS["k12_pop"], POP_PARAMS["k21_pop"],
            POP_PARAMS["omega_ka"], POP_PARAMS["omega_ke"], POP_PARAMS["omega_c0"],
            POP_PARAMS["omega_k12"], POP_PARAMS["omega_k21"], dose=DOSE)
        return [{"ka": ka[i], "ke": ke[i], "V": V[i], "k12": k12[i], "k21": k21[i]}
                for i in range(N)]
    if scenario.generating_model == "one_cpt_iv":
        ke, V = models.draw_one_cpt_iv_params(
            rng, N, POP_PARAMS["ke_pop"], POP_PARAMS["c0_pop"],
            POP_PARAMS["omega_ke"], POP_PARAMS["omega_c0"], dose=DOSE)
        return [{"ke": ke[i], "V": V[i]} for i in range(N)]
    raise ValueError(scenario.generating_model)


def true_curve(scenario: Scenario, params, t, delta=None):
    """True concentration curve for a patient.

    When ``delta`` is supplied it is a callable non-compartmental discrepancy
    (see ``models.make_gp_perturbation``); the sum is floored at zero so the
    true curve remains a valid concentration. Passing ``delta=None`` recovers
    the purely compartmental truth used by S1-S6.
    """
    if scenario.generating_model == "one_cpt_oral":
        c = models.one_cpt_oral(t, params["ka"], params["ke"], params["V"], DOSE)
    elif scenario.generating_model == "two_cpt_oral":
        c = models.two_cpt_oral(t, params["ka"], params["ke"], params["k12"],
                                params["k21"], params["V"], DOSE)
    elif scenario.generating_model == "one_cpt_iv":
        c = models.one_cpt_iv(t, params["ke"], params["V"], DOSE)
    else:
        raise ValueError(scenario.generating_model)
    if delta is not None:
        c = np.maximum(c + delta(t), 0.0)
    return c


def prior_mean_func(scenario: Scenario):
    """Mechanistically informed prior mean using the population parameters."""
    if scenario.fitted_mean == "one_cpt_oral":
        return lambda t: models.one_cpt_oral(t, POP_PARAMS["ka_pop"],
                                              POP_PARAMS["ke_pop"],
                                              DOSE / POP_PARAMS["c0_pop"], DOSE)
    if scenario.fitted_mean == "one_cpt_iv":
        return lambda t: models.one_cpt_iv(t, POP_PARAMS["ke_pop"],
                                             DOSE / POP_PARAMS["c0_pop"], DOSE)
    raise ValueError(scenario.fitted_mean)


def truth_summaries(scenario: Scenario, params, a, b, n_quad=64, delta=None):
    """Return true AUC[a:b], Cmax, Tmax for a patient under the scenario.

    ``delta`` must be the same realization used to generate that patient's
    observations, so that the ground truth and the data agree.
    """
    t_quad, w_quad = gp.gauss_legendre_nodes(a, b, n_quad)
    curve = true_curve(scenario, params, t_quad, delta)
    auc_true = float(w_quad @ curve)
    # Dense grid for the maximum
    t_grid = np.linspace(a, b, 4001)
    curve_grid = true_curve(scenario, params, t_grid, delta)
    cmax_true = float(curve_grid.max())
    tmax_true = float(t_grid[int(curve_grid.argmax())])
    return auc_true, cmax_true, tmax_true


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def run_scenario(scenario: Scenario, B=30, N=10, n_paths=1500,
                 seed=2024, verbose=True,
                 kernel="matern52", sigma_f_scale=0.5, sigma_eps_scale=0.2,
                 ell_fixed=4.0, prior_log_sd=0.2):
    """Generate B simulated datasets of N patients each, fit all six methods,
    and return a long-form DataFrame with one row per (dataset, patient,
    method, functional).
    """
    rng = np.random.default_rng(seed + SEED_OFFSET[scenario.name])

    a, b = TIME_WINDOW
    grid = scenario.grid
    mech_mean = prior_mean_func(scenario)

    rows = []
    for b_idx in range(B):
        params_list = draw_true_params(scenario, rng, N)
        for i, params in enumerate(params_list):
            # One discrepancy realization per patient, shared by the observed
            # data and the ground-truth functionals.
            delta = models.make_gp_perturbation(rng, scenario.tau,
                                                ell=scenario.ell_delta,
                                                window=TIME_WINDOW)
            true_y_grid = true_curve(scenario, params, grid, delta)
            y_obs = true_y_grid + rng.normal(0.0, SIGMA_EPS, size=len(grid))
            auc_true, cmax_true, tmax_true = truth_summaries(scenario, params, a, b,
                                                             delta=delta)

            # Run all methods on this patient
            method_runs = {}
            method_runs["NCA"] = methods.method_nca(grid, y_obs)
            method_runs["NCA + bootstrap"] = methods.method_nca_bootstrap(
                grid, y_obs, n_boot=200, rng=rng)
            if scenario.fitted_mean == "one_cpt_oral":
                method_runs["Parametric (correct cpt)"] = methods.method_parametric_1cpt_oral(
                    grid, y_obs, DOSE, a, b, rng=rng)
                method_runs["Parametric 2-cpt"] = methods.method_parametric_2cpt_oral(
                    grid, y_obs, DOSE, a, b, rng=rng)
            else:  # one_cpt_iv
                method_runs["Parametric (correct cpt)"] = methods.method_parametric_1cpt_iv(
                    grid, y_obs, DOSE, a, b, rng=rng)
                method_runs["Parametric 2-cpt"] = methods.method_parametric_2cpt_oral(
                    grid, y_obs, DOSE, a, b, rng=rng)
            # sigma_f_scale is deliberately left to the method's own
            # data-driven default: the zero-mean GP must represent the whole
            # curve, so the compartment-driven residual scale does not apply.
            method_runs["Pure GP"] = methods.method_pure_gp(
                grid, y_obs, a, b, n_paths=n_paths, rng=rng, kernel=kernel,
                sigma_eps_scale=sigma_eps_scale, ell_fixed=ell_fixed,
                prior_log_sd=prior_log_sd)
            method_runs["Mech.\\ centered GP"] = methods.method_mech_gp(
                grid, y_obs, mech_mean, a, b, n_paths=n_paths, rng=rng,
                kernel=kernel, sigma_f_scale=sigma_f_scale,
                sigma_eps_scale=sigma_eps_scale, ell_fixed=ell_fixed,
                prior_log_sd=prior_log_sd)

            for mname, mres in method_runs.items():
                rows.append({
                    "scenario": scenario.name,
                    "dataset": b_idx,
                    "patient": i,
                    "method": mname,
                    "auc_true": auc_true,
                    "auc_hat": mres["auc"],
                    "auc_lo": mres["auc_lo"],
                    "auc_hi": mres["auc_hi"],
                    "cmax_true": cmax_true,
                    "cmax_hat": mres["cmax"],
                    "cmax_lo": mres["cmax_lo"],
                    "cmax_hi": mres["cmax_hi"],
                })
        if verbose:
            print(f"  [{scenario.name}] dataset {b_idx + 1}/{B} done")

    df = pd.DataFrame(rows)
    return df


def summarize_estimates(estimates, lows, highs, truths):
    """Compute bias, RMSE, 95%-interval coverage and mean interval width."""
    e = np.asarray(estimates, dtype=float)
    lo = np.asarray(lows, dtype=float)
    hi = np.asarray(highs, dtype=float)
    t = np.asarray(truths, dtype=float)

    mask = np.isfinite(e) & np.isfinite(t)
    if mask.sum() == 0:
        return {"bias": np.nan, "rmse": np.nan, "coverage": np.nan,
                "width": np.nan}

    bias = float(np.mean(e[mask] - t[mask]))
    rmse = float(np.sqrt(np.mean((e[mask] - t[mask]) ** 2)))

    cov_mask = np.isfinite(lo) & np.isfinite(hi) & mask
    if cov_mask.sum() == 0:
        coverage = np.nan
        width = np.nan
    else:
        coverage = float(np.mean((t[cov_mask] >= lo[cov_mask]) &
                                  (t[cov_mask] <= hi[cov_mask])))
        width = float(np.mean(hi[cov_mask] - lo[cov_mask]))

    return {"bias": bias, "rmse": rmse, "coverage": coverage, "width": width}


def summarize_scenario(df, scenario_name=None):
    """Aggregate to one row per method with bias / RMSE / coverage / width for
    AUC and Cmax."""
    out = []
    df_scn = df if scenario_name is None else df[df.scenario == scenario_name]
    for m, g in df_scn.groupby("method"):
        s_auc = summarize_estimates(g["auc_hat"], g["auc_lo"], g["auc_hi"],
                                  g["auc_true"])
        s_cmax = summarize_estimates(g["cmax_hat"], g["cmax_lo"], g["cmax_hi"],
                                   g["cmax_true"])
        out.append({
            "scenario": scenario_name or df_scn["scenario"].iloc[0],
            "method": m,
            "auc_bias": s_auc["bias"], "auc_rmse": s_auc["rmse"],
            "auc_coverage": s_auc["coverage"], "auc_width": s_auc["width"],
            "cmax_bias": s_cmax["bias"], "cmax_rmse": s_cmax["rmse"],
            "cmax_coverage": s_cmax["coverage"], "cmax_width": s_cmax["width"],
        })
    return pd.DataFrame(out)
