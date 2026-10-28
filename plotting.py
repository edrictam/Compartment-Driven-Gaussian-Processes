"""Figures for the simulation and individualized oral-drug analyses.

The simulation notebook generates all four simulation figures with
make_fig2, make_fig3, make_fig4, and make_fig5. The warfarin and theophylline
notebooks use fit_oral_patient and plot_two_patients for the selected-patient
panels; those fits retain their own prediction window and random seed.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scenarios import (SCENARIOS, draw_true_params, true_curve,
                              prior_mean_func, truth_summaries, DOSE,
                              TIME_WINDOW, SIGMA_EPS)
import gp, methods, models
from models import one_cpt_oral

# Analysis settings match the production simulation in simulations.ipynb.
SIM_KERNEL = "matern52"
SIM_SIGMA_F = 0.6
SIM_ELL = 4.0


warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.2,
    "figure.dpi": 100,
})

OUTPUT_ROOT = Path(os.environ.get("PKGP_OUTPUT_DIR", Path(__file__).resolve().parent))
RESULTS_DIR = OUTPUT_ROOT / "results"
FIG_DIR = OUTPUT_ROOT / "figures"
os.makedirs(FIG_DIR, exist_ok=True)

METHOD_ORDER = [
    "NCA",
    "NCA + bootstrap",
    "Parametric (correct cpt)",
    "Parametric 2-cpt",
    "Pure GP",
    "Mech.\\ centered GP",
]
METHOD_DISPLAY = {
    "NCA": "NCA",
    "NCA + bootstrap": "NCA + boot",
    "Parametric (correct cpt)": "Param (correct)",
    "Parametric 2-cpt": "Param 2-cpt",
    "Pure GP": "Pure GP",
    "Mech.\\ centered GP": "Compartment-driven GP",
}
METHOD_COLORS = {
    "NCA": "#cccccc",
    "NCA + bootstrap": "#888888",
    "Parametric (correct cpt)": "#1f77b4",
    "Parametric 2-cpt": "#9467bd",
    "Pure GP": "#2ca02c",
    "Mech.\\ centered GP": "#d62728",
}
SCENARIOS_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]


# -- selected-patient oral-drug panels --------------------------------------

def fit_oral_patient(t_obs, y_obs, t_dose, wt, dose, ka, ke, vkg, ell_fixed,
                     t_end, seed=42):
    rng = np.random.default_rng(seed)
    V = vkg * wt
    def mf(t): return one_cpt_oral(np.asarray(t, float) - t_dose, ka, ke, V, dose)
    a, b = float(t_dose), float(max(t_dose + t_end, t_obs.max() + 12.0))
    res = methods.method_mech_gp(t_obs, y_obs, mf, a, b, n_paths=2000, rng=rng,
                                  kernel='matern52', sigma_f_scale=1.5,
                                  sigma_eps_scale=0.2, ell_fixed=ell_fixed,
                                  prior_log_sd=0.2)
    res['_mean_func'] = mf
    res['_t_obs'] = t_obs; res['_y_obs'] = y_obs
    res['_wt'] = wt; res['_dose'] = dose
    return res


def plot_two_patients(fits, ids, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, pid in zip(axes, ids):
        r = fits[pid]
        grid = r['_paths_grid']; mu = r['_mu_grid']; sd = np.sqrt(r['_K_grid_diag'])
        ax.fill_between(grid, mu - 1.96*sd, mu + 1.96*sd, color='#d62728', alpha=0.18,
                        label='posterior 95% band')
        ax.plot(grid, mu, color='#d62728', lw=1.6, label='posterior mean')
        ax.scatter(r['_t_obs'], r['_y_obs'], color='black', s=30, zorder=5, label='observed')
        ax.set_title(f"Patient {pid}  (WT={r['_wt']:.0f} kg, dose={r['_dose']:.0f} mg, {len(r['_t_obs'])} obs)")
        ax.set_xlabel('time (hr)'); ax.set_ylabel('mg/L')
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig); print(f'saved {out_path}')

# -- simulation comparison panels -----------------------------------------


def grouped_bar(ax, df, value_col, ylabel, ref=None, log=False):
    pivot = df.pivot(index="scenario", columns="method", values=value_col)
    pivot = pivot.reindex(SCENARIOS_ORDER)[METHOD_ORDER]
    x = np.arange(len(pivot.index))
    n_m = len(METHOD_ORDER)
    width = 0.8 / n_m
    for i, m in enumerate(METHOD_ORDER):
        vals = pivot[m].values.astype(float)
        # Replace NaN bars by 0 height + a hatched marker
        vals_plot = np.where(np.isnan(vals), 0.0, vals)
        ax.bar(x + (i - (n_m - 1) / 2) * width, vals_plot, width,
               label=METHOD_DISPLAY[m], color=METHOD_COLORS[m],
               edgecolor="white", linewidth=0.4)
    if ref is not None:
        ax.axhline(ref, color="black", linestyle="--", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("symlog", linthresh=1)
    ax.grid(axis="y", alpha=0.3, lw=0.5)


def make_panel_figure(df, panels, out_path, figsize=(11, 7.2), title=None):
    n = len(panels)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=figsize)
    axes = np.atleast_2d(axes).reshape(nrows, 2)
    for idx, (col, ylabel, ref, log) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        grouped_bar(ax, df, col, ylabel, ref=ref, log=log)
        ax.set_title(ylabel)
    for k in range(n, nrows * 2):
        axes[k // 2, k % 2].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=min(len(METHOD_ORDER), 3),
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    if title:
        fig.suptitle(title, y=1.07)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def make_fig3(df):
    panels = [
        ("auc_bias", "AUC bias (mg$\\cdot$hr/L)", 0.0, True),
        ("auc_rmse", "AUC RMSE (mg$\\cdot$hr/L)", None, True),
        ("auc_coverage", "AUC 95% coverage", 0.95, False),
        ("auc_width", "AUC 95% interval width", None, True),
    ]
    make_panel_figure(df, panels,
                      os.path.join(FIG_DIR, "fig3_auc_results.pdf"))


def make_fig4(df):
    panels = [
        ("cmax_bias", "$C_{\\max}$ bias (mg/L)", 0.0, False),
        ("cmax_rmse", "$C_{\\max}$ RMSE (mg/L)", None, False),
        ("cmax_coverage", "$C_{\\max}$ 95% coverage", 0.95, False),
        ("cmax_width", "$C_{\\max}$ 95% interval width", None, False),
    ]
    make_panel_figure(df, panels,
                      os.path.join(FIG_DIR, "fig4_cmax_results.pdf"),
                      figsize=(11, 10.5))


# -- fig2: single-patient posterior paths under S1 -------------------------

def make_fig2(seed=42):
    sc = SCENARIOS["S1"]
    rng = np.random.default_rng(seed)
    params = draw_true_params(sc, rng, 1)[0]
    grid = sc.grid
    y_obs = true_curve(sc, params, grid) + rng.normal(0, SIGMA_EPS, len(grid))
    a, b = TIME_WINDOW
    mech = prior_mean_func(sc)
    res = methods.method_mech_gp(grid, y_obs, mech, a, b, n_paths=2000, rng=rng,
                                 kernel=SIM_KERNEL, sigma_f_scale=SIM_SIGMA_F,
                                 sigma_eps_scale=0.2, ell_fixed=SIM_ELL,
                                 prior_log_sd=0.2)

    auc_t, cmax_t, _ = truth_summaries(sc, params, a, b)
    t_dense = np.linspace(a, b, 600)
    true_dense = true_curve(sc, params, t_dense)

    paths = res["_paths"]
    t_grid_p = res["_paths_grid"]
    mu_p = res["_mu_grid"]

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.2],
                          hspace=0.55, wspace=0.3)
    ax0 = fig.add_subplot(gs[:, 0])
    n_show = min(20, paths.shape[1])
    for s in range(n_show):
        ax0.plot(t_grid_p, paths[:, s], color="lightgray", alpha=0.6, lw=0.7)
    ax0.plot(t_dense, true_dense, color="#1f77b4", lw=1.2, alpha=0.6,
              label="true curve")
    ax0.plot(t_grid_p, mu_p, color="black", lw=1.6, label="posterior mean")
    ax0.scatter(grid, y_obs, color="black", s=24, zorder=5, label="observed")
    ax0.set_xlabel("time (hours)")
    ax0.set_ylabel("concentration (mg/L)")
    ax0.set_title("Posterior latent curve")
    ax0.legend(loc="best")

    # Posterior paths -> AUC samples and Cmax samples
    t_q, w_q = gp.gauss_legendre_nodes(a, b, 64)
    mu_q, K_q = gp.gp_posterior(grid, y_obs, t_q, mech(grid), mech(t_q),
                                 *res["_hyper"], kernel=SIM_KERNEL)
    auc_mean, auc_var = gp.auc_posterior_gaussian(mu_q, K_q, w_q)
    auc_sd = np.sqrt(auc_var)
    # AUC samples from posterior paths on the path grid for kernel density
    auc_samples = (paths.T @ np.ones_like(t_grid_p)) * (b - a) / len(t_grid_p)
    # That's a rough trapezoidal; use Gauss-Legendre instead for cleaner samples
    paths_q, _ = gp.gp_posterior(grid, y_obs, t_q, mech(grid), mech(t_q),
                                  *res["_hyper"], kernel=SIM_KERNEL)
    paths_qsamples = gp.sample_posterior_paths(paths_q, K_q, paths.shape[1], rng)
    auc_samples = paths_qsamples.T @ w_q
    cmax_samples = gp.cmax_posterior_samples(paths)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(auc_samples, bins=30, density=True, color="#d62728", alpha=0.6,
             edgecolor="white")
    gx = np.linspace(auc_samples.min(), auc_samples.max(), 200)
    pdf = np.exp(-0.5 * ((gx - auc_mean) / auc_sd) ** 2) / (auc_sd * np.sqrt(2 * np.pi))
    ax1.plot(gx, pdf, color="black", lw=1.4, label="implied Gaussian")
    ax1.axvline(auc_t, color="#1f77b4", lw=1.2, label="truth")
    ax1.set_title("Posterior of AUC")
    ax1.set_xlabel("AUC (mg$\\cdot$hr/L)")
    ax1.set_yticks([])
    ax1.legend(loc="best", fontsize=8)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.hist(cmax_samples, bins=30, density=True, color="#d62728", alpha=0.6,
             edgecolor="white")
    ax2.axvline(cmax_t, color="#1f77b4", lw=1.2, label="truth")
    ax2.set_title("Posterior of $C_{\\max}$")
    ax2.set_xlabel("$C_{\\max}$ (mg/L)")
    ax2.set_yticks([])
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig2_posterior_paths.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# -- fig5: sparse-sampling case for S4 -------------------------------------

def make_fig5(seed=23):
    sc = SCENARIOS["S4"]
    rng = np.random.default_rng(seed)
    # Draw an illustrative patient with a peak between 1.4 and 3.6 hours.
    for _ in range(20):
        params = draw_true_params(sc, rng, 1)[0]
        grid = sc.grid  # {1, 2, 4, 8, 16} hours
        true_grid = true_curve(sc, params, grid)
        _, cmax_t, tmax_t = truth_summaries(sc, params, *TIME_WINDOW)
        # Retain the selection criterion: avoid peaks close to 1 or 4 hours.
        # The full sparse design also includes the 2-hour observation.
        if 1.4 < tmax_t < 3.6 and abs(tmax_t - 1.0) > 0.4 and abs(tmax_t - 4.0) > 0.4:
            break
    y_obs = true_grid + rng.normal(0, SIGMA_EPS, len(grid))
    a, b = TIME_WINDOW
    mech = prior_mean_func(sc)

    t_dense = np.linspace(a, b, 600)
    true_dense = true_curve(sc, params, t_dense)

    # Mech GP
    mech_res = methods.method_mech_gp(grid, y_obs, mech, a, b, n_paths=1000, rng=rng,
                                      kernel=SIM_KERNEL, sigma_f_scale=SIM_SIGMA_F,
                                      sigma_eps_scale=0.2, ell_fixed=SIM_ELL,
                                      prior_log_sd=0.2)
    mech_mu = mech_res["_mu_grid"]
    mech_sd = np.sqrt(mech_res["_K_grid_diag"])
    mech_t = mech_res["_paths_grid"]

    # Pure GP
    pure_res = methods.method_pure_gp(grid, y_obs, a, b, n_paths=1000, rng=rng,
                                      kernel=SIM_KERNEL, sigma_eps_scale=0.2,
                                      ell_fixed=SIM_ELL, prior_log_sd=0.2)
    pure_mu = pure_res["_mu_grid"]
    pure_sd = np.sqrt(pure_res["_K_grid_diag"])
    pure_t = pure_res["_paths_grid"]

    # Parametric 1-cpt oral
    ka, ke, V, _ = methods._fit_1cpt_oral_nls(grid, y_obs, DOSE)
    param_curve = models.one_cpt_oral(t_dense, ka, ke, V, DOSE)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(mech_t, mech_mu - 1.96 * mech_sd, mech_mu + 1.96 * mech_sd,
                    color="#d62728", alpha=0.18,
                    label="Compartment-driven GP 95% band")
    ax.plot(mech_t, mech_mu, color="#d62728", lw=1.6,
            label="Compartment-driven GP mean")
    ax.plot(pure_t, pure_mu, color="#2ca02c", lw=1.2, linestyle=":",
            label="Pure GP mean")
    ax.plot(t_dense, param_curve, color="#1f77b4", lw=1.2, linestyle="--",
            label="Parametric 1-cpt fit")
    ax.plot(t_dense, true_dense, color="gray", lw=1.0, label="true curve")
    ax.scatter(grid, y_obs, color="black", s=44, zorder=5, label="observed")
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("concentration (mg/L)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig5_sparse_case.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
