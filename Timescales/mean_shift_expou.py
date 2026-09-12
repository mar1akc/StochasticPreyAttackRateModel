"""
Small-noise (Lyapunov) theory for the exp-OU-driven role-reversal model.

Linearize the augmented system around (E*(g_bar), U = 0) and write p = P - E*:

    dp = [J0 p + j U] dt + O(2),      dU = -alpha U dt + sigma dB,

where J0 = D_P F at E* and j = d F / d U at U = 0.  Only the y1-equation
depends on G, through the term -y1 G x with G = g_min + Delta e^U, so

    j = (0, -Delta x* y1*, 0)^T,      Delta = g_bar - g_min.

Stationary second moments (leading order in sigma):

    Sigma_pu := E[p U]  solves   (J0 - alpha I) Sigma_pu = -v j,   v = sigma^2/(2 alpha)
    Sigma_pp := E[p p^T] solves  J0 Sigma + Sigma J0^T = v ( j Jj^T + Jj j^T ),
                                 Jj = (J0 - alpha I)^{-1} j.

The O(sigma^2) shift of the stationary mean then follows from averaging the
second-order Taylor terms of the drift:

    0 = J0 E[p] + q,      E[p] = -J0^{-1} q,

with, component by component (the y2-equation is linear, so q_3 = 0),

    q_1 = -a S_xx + s S_x,y1 - b S_x,y2
    q_2 =  k S_x,y2 - g_bar S_x,y1
          - Delta ( x* Sigma_pu[y1] + y1* Sigma_pu[x] + x* y1* v / 2 ).

The last group collects the three ways the driver enters at second order:
the p-U correlations and the convexity of the exponential map, the latter
being exactly the excess E[G] - g_bar = Delta (e^{v/2} - 1) ~ Delta v / 2.
"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from sde import Params, E_star, jacobian, g_min_default


def j_vector(p: Params, g_bar, eq=None, g_min=None):
    """dF/dU at U = 0:  only the y1-component is non-zero."""
    if eq is None:
        eq = E_star(p.with_(g=g_bar))
    if g_min is None:
        g_min = g_min_default(p)
    x_s, y1_s, _ = eq
    delta = g_bar - g_min
    return np.array([0.0, -delta * x_s * y1_s, 0.0])


def second_moments(p: Params, g_bar, sigma, alpha, g_min=None):
    """Stationary Sigma_pp and Sigma_pu of the linearized system."""
    if g_min is None:
        g_min = g_min_default(p)
    pg = p.with_(g=g_bar)
    eq = E_star(pg)
    J0 = jacobian(eq, pg)
    j = j_vector(p, g_bar, eq=eq, g_min=g_min)

    v = sigma**2 / (2.0 * alpha)
    Jj = np.linalg.solve(J0 - alpha * np.eye(3), j)      # (J0 - alpha I)^{-1} j

    Sigma_pu = -v * Jj
    Jj_jT = np.outer(Jj, j)
    Q = v * (Jj_jT + Jj_jT.T)
    Sigma_pp = solve_continuous_lyapunov(J0, Q)
    Sigma_pp = 0.5 * (Sigma_pp + Sigma_pp.T)             # clean up round-off
    return Sigma_pp, Sigma_pu, eq, J0


def mean_shifts(p: Params, g_bar, sigma, alpha, g_min=None,
                require_stable=True):
    """
    O(sigma^2) shift of the stationary mean away from the deterministic E*.

    Returns (shift, eq) with shift = E[P] - E*, or None when the deterministic
    equilibrium is unstable (the small-noise expansion about E* does not apply
    there) or when sigma = 0.
    """
    if sigma == 0.0:
        return None
    if g_min is None:
        g_min = g_min_default(p)

    Spp, Spu, eq, J0 = second_moments(p, g_bar, sigma, alpha, g_min=g_min)

    if require_stable and np.max(np.real(np.linalg.eigvals(J0))) >= 0.0:
        return None

    x_s, y1_s, _ = eq
    v = sigma**2 / (2.0 * alpha)
    delta = g_bar - g_min

    q = np.zeros(3)
    q[0] = -p.a * Spp[0, 0] + p.s * Spp[0, 1] - p.b * Spp[0, 2]
    q[1] = (p.k * Spp[0, 2] - g_bar * Spp[0, 1]
            - delta * (x_s * Spu[1] + y1_s * Spu[0] + 0.5 * v * x_s * y1_s))

    shift = -np.linalg.solve(J0, q)
    return shift, eq
