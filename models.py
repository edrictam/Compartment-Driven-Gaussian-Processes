"""Compartmental pharmacokinetic mean functions.

Each function returns the plasma concentration (mg/L) at the requested times for
a given dose (mg) and rate constants. Units assumed throughout:
    time  in hours
    rates in 1/hour
    volumes in L
    dose in mg
"""

import numpy as np


# ---------------------------------------------------------------------------
# One-compartment models
# ---------------------------------------------------------------------------

def one_cpt_oral(t, ka, ke, V, dose):
    """One-compartment oral absorption.

    m(t) = (D * ka) / (V * (ka - ke)) * (exp(-ke t) - exp(-ka t))
    Handles the degenerate ka == ke case as a limit.
    """
    t = np.asarray(t, dtype=float)
    if np.isclose(ka, ke):
        return (dose / V) * ka * t * np.exp(-ke * t)
    return (dose * ka) / (V * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def one_cpt_iv(t, ke, V, dose):
    """One-compartment IV bolus.

    m(t) = (D / V) * exp(-ke t)
    """
    t = np.asarray(t, dtype=float)
    return (dose / V) * np.exp(-ke * t)


def one_cpt_oral_cmax_tmax(ka, ke, V, dose):
    """Closed-form tmax and Cmax for the 1-compartment oral model on [0, inf)."""
    if np.isclose(ka, ke):
        tmax = 1.0 / ke
    else:
        tmax = np.log(ka / ke) / (ka - ke)
    cmax = one_cpt_oral(tmax, ka, ke, V, dose)
    return tmax, cmax


def one_cpt_iv_cmax_tmax(ke, V, dose):
    """Cmax for 1-compartment IV bolus is at t = 0."""
    return 0.0, dose / V


# ---------------------------------------------------------------------------
# Two-compartment oral model (closed-form)
# ---------------------------------------------------------------------------

def two_cpt_oral(t, ka, ke, k12, k21, V, dose):
    """Two-compartment oral absorption (closed-form plasma concentration).

    Compartments: depot A_d, central A_c, peripheral A_p with first-order rates
    and elimination only from the central compartment. The plasma
    concentration C(t) = A_c(t)/V admits the closed form (Gibaldi & Perrier)

        C(t) = (D ka / V) * [
            (k21 - alpha)/((ka - alpha)(beta - alpha)) exp(-alpha t)
          + (k21 - beta) /((ka - beta) (alpha - beta)) exp(-beta t)
          + (k21 - ka)  /((alpha - ka)(beta - ka))     exp(-ka t) ]

    where alpha, beta are the roots of lambda^2 + (ke+k12+k21) lambda + ke k21 = 0.
    Degenerate cases (e.g., ka == alpha) are perturbed by a small epsilon.
    """
    t = np.asarray(t, dtype=float)
    s = ke + k12 + k21
    disc = s * s - 4.0 * ke * k21
    disc = max(disc, 0.0)
    sq = np.sqrt(disc)
    alpha = 0.5 * (s + sq)
    beta = 0.5 * (s - sq)

    # Guard against accidental degeneracies in the denominators.
    eps = 1e-9
    if abs(ka - alpha) < eps: ka = ka + eps
    if abs(ka - beta) < eps:  ka = ka + eps
    if abs(alpha - beta) < eps: alpha = alpha + eps

    coef_a = (k21 - alpha) / ((ka - alpha) * (beta - alpha))
    coef_b = (k21 - beta)  / ((ka - beta)  * (alpha - beta))
    coef_k = (k21 - ka)    / ((alpha - ka) * (beta - ka))

    return (dose * ka / V) * (
        coef_a * np.exp(-alpha * t)
        + coef_b * np.exp(-beta * t)
        + coef_k * np.exp(-ka * t)
    )


# ---------------------------------------------------------------------------
# Population samplers (Gaussian, not log-Gaussian, with small SDs)
#
# Between-patient variability is placed directly on the concentration scale,
# through the initial concentration c0, rather than on an apparent volume of
# distribution. The closed-form curves above are parameterized by (V, dose),
# so these samplers return V = dose / c0; c0 is the quantity actually drawn
# and the only one the simulation reports.
# ---------------------------------------------------------------------------

def draw_one_cpt_oral_params(rng, n, ka_pop=1.5, ke_pop=0.1, c0_pop=10.0,
                             omega_ka=0.3, omega_ke=0.02, omega_c0=2.0,
                             dose=300.0):
    ka = rng.normal(ka_pop, omega_ka, size=n)
    ke = rng.normal(ke_pop, omega_ke, size=n)
    c0 = rng.normal(c0_pop, omega_c0, size=n)
    # Clip away from degeneracy / negative values; with the chosen SDs, the
    # tails contributing to such draws are negligibly small.
    ka = np.clip(ka, 0.5, None)
    ke = np.clip(ke, 0.02, None)
    c0 = np.clip(c0, 2.0, None)
    return ka, ke, dose / c0


def draw_one_cpt_iv_params(rng, n, ke_pop=0.1, c0_pop=10.0,
                           omega_ke=0.02, omega_c0=2.0, dose=300.0):
    ke = rng.normal(ke_pop, omega_ke, size=n)
    c0 = rng.normal(c0_pop, omega_c0, size=n)
    ke = np.clip(ke, 0.02, None)
    c0 = np.clip(c0, 2.0, None)
    return ke, dose / c0


def draw_two_cpt_oral_params(rng, n, ka_pop=1.5, ke_pop=0.1, c0_pop=10.0,
                             k12_pop=0.2, k21_pop=0.1,
                             omega_ka=0.3, omega_ke=0.02, omega_c0=2.0,
                             omega_k12=0.04, omega_k21=0.02, dose=300.0):
    ka  = np.clip(rng.normal(ka_pop,  omega_ka,  size=n), 0.5, None)
    ke  = np.clip(rng.normal(ke_pop,  omega_ke,  size=n), 0.02, None)
    c0  = np.clip(rng.normal(c0_pop,  omega_c0,  size=n), 2.0, None)
    k12 = np.clip(rng.normal(k12_pop, omega_k12, size=n), 0.02, None)
    k21 = np.clip(rng.normal(k21_pop, omega_k21, size=n), 0.02, None)
    return ka, ke, dose / c0, k12, k21


# ---------------------------------------------------------------------------
# Non-compartmental discrepancy
# ---------------------------------------------------------------------------

_CHOL_CACHE = {}


def make_gp_perturbation(rng, tau, ell=4.0, window=(0.0, 24.0), n_ref=481,
                         kernel="matern52"):
    """Draw a smooth random discrepancy delta(t) ~ GP(0, tau^2 k(.,.)).

    Returns a callable evaluable at arbitrary times, so that a single
    realization can be shared by the observation grid, the quadrature nodes
    used for the true AUC, and the dense grid used for the true C_max.

    The path is drawn on a fine reference grid and cubic-splined; with
    ``n_ref`` points over a 24 hr window the spacing is ~0.05 hr, far finer
    than any length scale of interest, so interpolation error is negligible.
    The Cholesky factor depends only on the grid and the kernel, not on
    ``tau``, so it is computed once and cached across patients.
    """
    from scipy.interpolate import CubicSpline
    from gp import kernel_matrix

    if tau <= 0.0:
        return None
    key = (n_ref, float(ell), kernel, tuple(window))
    if key not in _CHOL_CACHE:
        ref = np.linspace(window[0], window[1], n_ref)
        K = kernel_matrix(ref, ref, 1.0, ell, kernel=kernel)
        _CHOL_CACHE[key] = (ref, np.linalg.cholesky(K + 1e-8 * np.eye(n_ref)))
    ref, L = _CHOL_CACHE[key]
    return CubicSpline(ref, tau * (L @ rng.standard_normal(n_ref)))

