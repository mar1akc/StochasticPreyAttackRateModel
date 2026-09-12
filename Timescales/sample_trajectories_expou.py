"""
Sample trajectories illustrating the noise-induced dynamics (Fig. `fig:sample_traj`).

Three values of the OU mean (median) g_bar straddle the deterministic Hopf
bifurcation g_1 ~ 0.4385:

    (a) g_bar = 0.42 -- below g_1 (deterministic limit cycle)
    (b) g_bar = 0.45 -- just above g_1 (deterministically stable focus)
    (c) g_bar = 0.50 -- comfortably stable deterministically

For each we show the deterministic run (v = 0), the stochastic run
(alpha = 0.133, v = sigma_U^2/(2 alpha) = 0.00125, i.e. sigma_U ~ 0.0182)
together with the realized driver G(t), and the (x, y2) phase-plane
projection.  The correlation time 1/alpha ~ 7.5 is close to 1/omega_0, the
resonance condition.  Each phase panel is scaled to its own orbits, so the
shape of the noise-induced cloud is legible at every g_bar (pass
shared_phase_limits=True for common limits instead).  The stochastic-mean
equilibrium predicted by the small-noise/Lyapunov theorem is overlaid
wherever E* is stable.

Run as a script, or import `make_figure` from a notebook.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from sde import Params, E_star, jacobian, g_hopf, g_min_default, OUNoise, \
                simulate_exp_ou_sde
from mean_shift_expou import mean_shifts

# --------------------------------------------------------------------------
# Configuration -- parameter values of Li, Liu & Wei (2022), as in the paper.
# --------------------------------------------------------------------------
P = Params(r=0.4, a=0.01, k=0.3, b=0.4, D=0.4, v1=0.2, v2=0.2, s=0.2)

# The driver is parametrized by its mean-reversion rate alpha and the
# stationary variance v = sigma_U^2 / (2 alpha) of the latent process U, so
# that the driver amplitude std(U) = sqrt(v) is held fixed and alpha alone
# sets the correlation time 1/alpha.  alpha = 0.133 is close to the
# resonance value: it matches the natural frequency omega_0 ~ 0.128 of the
# linearization at E*(g_bar = 0.45).
ALPHA        = 0.133                            # 1/alpha ~ 7.5
VAR_U        = 0.00125                          # v = sigma_U^2 / (2 alpha)
SIGMA        = float(np.sqrt(2.0*ALPHA*VAR_U))  # sigma_U ~ 0.018235
T_END        = 1500.0
DT           = 0.01
STORE_EVERY  = 5
SEED         = 1
T_BURN_PHASE = 500.0      # burn-in discarded in the phase-plane panels

CASES = [
    (0.42, r"below $g_1$"),
    (0.45, r"just above $g_1$"),
    (0.50, r"deterministically stable"),
]

# Legend text for the small-noise prediction.  Keep in sync with the theorem
# number in the manuscript (\ref{thm:mean-shift}).
THM_LABEL = "stoch. mean (Thm. 5)"

OUTDIR_PNG = os.environ.get("EXPOU_PNG_DIR", ".")
OUTDIR_PDF = os.environ.get("EXPOU_PDF_DIR", ".")

LEG_FS, LAB_FS, TIT_FS, TICK_FS = 10, 12, 12, 10

# Fraction of the middle-panel height occupied by the driver trace G(t).
G_BAND = 0.45


# --------------------------------------------------------------------------
def run_cases(p=P, cases=CASES, sigma=SIGMA, alpha=ALPHA, t_end=T_END, dt=DT,
              store_every=STORE_EVERY, seed=SEED):
    """Simulate the deterministic and stochastic run for every case."""
    g_min = g_min_default(p)
    runs = []
    for g_bar, label in cases:
        eq = E_star(p.with_(g=g_bar))
        ic = tuple(eq * np.array([1.3, 0.8, 0.8]))     # perturbed E*
        common = dict(t_end=t_end, dt=dt, ic=ic, seed=seed,
                      store_every=store_every)
        ou_det = OUNoise(g_bar=g_bar, sigma=0.0, alpha=alpha, g_min=g_min)
        ou_noi = OUNoise(g_bar=g_bar, sigma=sigma, alpha=alpha, g_min=g_min)
        det = simulate_exp_ou_sde(p, ou_det, **common)
        noi = simulate_exp_ou_sde(p, ou_noi, **common)
        runs.append(dict(g_bar=g_bar, label=label, eq=eq, det=det, noi=noi,
                         ou=ou_noi))
    return runs


def phase_limits(runs, t_burn=T_BURN_PHASE, pad=0.08, top_pad=0.45,
                 shared=False):
    """
    (x, y2) phase-plane limits, one pair per run.

    By default each panel is scaled to its own orbits, so that the shape of
    the noise-induced cloud is legible at every g_bar; the price is that the
    axis ranges differ across rows.  With shared=True all rows instead get
    the envelope of every orbit drawn (set by the widest cloud, g_bar = 0.42),
    which makes the clouds directly comparable in size but shrinks the two
    stable rows to small blobs.

    `top_pad` leaves headroom above the orbit for the legend.
    """
    def env(run_list):
        xs, ys = [], []
        for run in run_list:
            for t, X in (run["det"], run["noi"]):
                m = t > t_burn
                xs.append(X[m, 0])
                ys.append(X[m, 2])
        x, y = np.concatenate(xs), np.concatenate(ys)
        dx = max(x.max() - x.min(), 1e-9)
        dy = max(y.max() - y.min(), 1e-9)
        return ((x.min() - pad*dx, x.max() + pad*dx),
                (y.min() - pad*dy, y.max() + top_pad*dy))

    if shared:
        return [env(runs)] * len(runs)
    return [env([run]) for run in runs]


# --------------------------------------------------------------------------
def make_figure(p=P, runs=None, sigma=SIGMA, alpha=ALPHA,
                t_burn=T_BURN_PHASE, g1=None, shared_phase_limits=False):
    if runs is None:
        runs = run_cases(p=p, sigma=sigma, alpha=alpha)
    if g1 is None:
        g1 = g_hopf(p)

    # The driver is reported through v = sigma_U^2/(2 alpha), the stationary
    # variance of the latent process U, rather than through sigma_U itself.
    var_u = sigma**2 / (2.0*alpha)
    lab_det = r"$v = 0$"
    lab_noi = rf"$v = {var_u:g}$"

    pp_lims = phase_limits(runs, t_burn=t_burn, shared=shared_phase_limits)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12),
                             gridspec_kw={"width_ratios": [2, 2, 1.6]})

    for row, run in enumerate(runs):
        g_bar, label, eq = run["g_bar"], run["label"], run["eq"]
        t_d, X_d = run["det"]
        t_n, X_n = run["noi"]
        ou = run["ou"]

        # ---- column 0: deterministic time series -------------------------
        ax = axes[row, 0]
        ax.plot(t_d, X_d[:, 0], "r-", lw=0.9, label=r"prey $x$")
        ax.plot(t_d, X_d[:, 2], "b-", lw=0.9, label=r"adult pred. $y_2$")
        ax.set_title(rf"$\bar g = {g_bar}$ ({label}), {lab_det}",
                     fontsize=TIT_FS)
        ax.set_xlabel("time", fontsize=LAB_FS)
        ax.set_ylabel("population", fontsize=LAB_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.legend(loc="upper right", fontsize=LEG_FS, framealpha=0.9)
        ax.grid(alpha=0.3)
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y0 + 1.15*(y1 - y0))          # headroom for the legend

        # ---- column 1: stochastic time series + driver G(t) --------------
        ax = axes[row, 1]
        l_prey, = ax.plot(t_n, X_n[:, 0], "r-", lw=0.6, label=r"prey $x$")
        l_pred, = ax.plot(t_n, X_n[:, 2], "b-", lw=0.6,
                          label=r"adult pred. $y_2$")
        ax.set_xlabel("time", fontsize=LAB_FS)
        ax.set_ylabel("population", fontsize=LAB_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        l_g, = ax2.plot(t_n, X_n[:, 3], "-", color="0.4", lw=0.6, alpha=0.7,
                        label=r"$G(t)$ (right axis)")
        l_g1 = ax2.axhline(g1, ls="--", color="gray", lw=1.1,
                           label=rf"$g_1 = {g1:.4f}$")
        ax2.set_ylabel(r"$G(t)$", color="0.3", fontsize=LAB_FS)
        ax2.tick_params(axis="y", labelcolor="0.3", labelsize=TICK_FS)
        # Confine the driver (and g_1) to the lower band of the panel, so the
        # gray trace does not cover the population curves.
        gl = min(X_n[:, 3].min(), g1)
        gh = max(X_n[:, 3].max(), g1)
        span = max(gh - gl, 1e-12)
        ax2.set_ylim(gl - 0.10*span, gl + span/G_BAND)

        handles = [l_prey, l_pred, l_g, l_g1]
        ax.legend(handles, [h.get_label() for h in handles], loc="upper left",
                  fontsize=LEG_FS, framealpha=0.9, ncol=2)
        ax.set_title(
            rf"$\bar g = {g_bar}$, $\alpha = {alpha}$, {lab_noi}, "
            rf"std$\,G = {ou.stationary_std_G(p):.4f}$", fontsize=TIT_FS)
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y0 + 1.25*(y1 - y0))

        # ---- column 2: (x, y2) phase-plane projection --------------------
        ax = axes[row, 2]
        m_d, m_n = t_d > t_burn, t_n > t_burn
        ax.plot(X_n[m_n, 0], X_n[m_n, 2], "g-", lw=0.4, alpha=0.55,
                label=lab_noi, zorder=2)
        ax.plot(X_d[m_d, 0], X_d[m_d, 2], "b-", lw=0.8, alpha=0.9,
                label=lab_det, zorder=3)
        # Open star so the (nearby) theoretical mean stays visible inside it.
        ax.plot([eq[0]], [eq[2]], "r*", ms=18, mfc="none", mew=1.8,
                label=r"det. $E^*$", zorder=5)

        ms = mean_shifts(p, g_bar, sigma, alpha)
        if ms is not None:
            shift, eq_th = ms
            ax.plot([eq_th[0] + shift[0]], [eq_th[2] + shift[2]], "kP", ms=8,
                    mfc="yellow", mew=1.2, label=THM_LABEL, zorder=6)

        ax.set_xlabel(r"prey $x$", fontsize=LAB_FS)
        ax.set_ylabel(r"adult pred. $y_2$", fontsize=LAB_FS)
        ax.tick_params(labelsize=TICK_FS)
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(l) for l in
                 [lab_det, lab_noi, r"det. $E^*$"]
                 if l in labels] + \
                ([labels.index(THM_LABEL)] if THM_LABEL in labels else [])
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  fontsize=LEG_FS, loc="upper right", framealpha=0.9)
        ax.set_title(rf"$\bar g = {g_bar}$", fontsize=TIT_FS)
        ax.grid(alpha=0.3)
        ax.set_xlim(*pp_lims[row][0])
        ax.set_ylim(*pp_lims[row][1])

    fig.tight_layout()
    return fig, runs


def save(fig, stem="sample_trajectories_expou"):
    os.makedirs(OUTDIR_PNG, exist_ok=True)
    os.makedirs(OUTDIR_PDF, exist_ok=True)
    png = os.path.join(OUTDIR_PNG, stem + ".png")
    pdf = os.path.join(OUTDIR_PDF, stem + ".pdf")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    return png, pdf


if __name__ == "__main__":
    fig, runs = make_figure()
    print("saved:", *save(fig))
    plt.close(fig)
