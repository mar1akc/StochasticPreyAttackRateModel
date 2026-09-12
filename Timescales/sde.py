"""
Stochastic predator-prey model with role reversal, in which the prey's attack
rate on juvenile predators is an *exponential* Ornstein-Uhlenbeck (exp-OU)
process supported on (g_min, infinity).

State (4D):    Z = (x, y1, y2, U),   reported as (x, y1, y2, G)

Dynamics:
    dx  = x*(r - a*x + s*y1 - b*y2) dt
    dy1 = [k*x*y2 - y1*(G(U)*x + D + v1)] dt
    dy2 = (D*y1 - v2*y2) dt
    dU  = -alpha*U dt + sigma_U dB(t)

with

    G(U) = g_min + Delta * exp(U),      Delta = g_bar - g_min > 0.

The latent driver U is an Ornstein-Uhlenbeck process with fixed point 0 and
stationary law N(0, v),  v = sigma_U^2 / (2*alpha),  correlation time 1/alpha.
Consequently G - g_min is lognormal, G(U) > g_min almost surely for all t, and

    E[G]    = g_min + Delta * exp(v/2)
    Var(G)  = Delta^2 * exp(v) * (exp(v) - 1).

Note that g_bar is the *fixed point* of the driver, equivalently the median of
the stationary law of G -- not its mean.  The mean exceeds g_bar by
Delta*(exp(v/2) - 1) = O(v).

The lower barrier g_min = s*k/b is the value above which the deterministic
model is guaranteed to have bounded solutions for all time; a driver with
support on all of R (in particular a plain OU process on g) enters {g < g_min}
with probability one and admits no global existence theorem.

Integration
-----------
* U:            Euler-Maruyama,   U_{n+1} = U_n - alpha*U_n*dt + sigma_U*sqrt(dt)*xi.
* populations:  classical four-stage, fourth-order Runge-Kutta, using G(U_n)
                for stage 1, G at the interpolated midpoint U_{n+1/2} =
                (U_n + U_{n+1})/2 for stages 2 and 3, and G(U_{n+1}) for
                stage 4.

Because the populations carry no noise of their own, RK4 integrates them to
fourth order along the realized driver path; the overall accuracy in the
driver is set by Euler-Maruyama.

To keep populations strictly positive we integrate in log coordinates by
default (log_state=True).
"""

import numpy as np
from dataclasses import dataclass, replace
from typing import Optional


DT_DEFAULT = 1e-3          # timestep used throughout


# ---------------------------------------------------------------------------
# Deterministic model.
#
# If model.py is importable we use it, so that this module and the rest of the
# project always share one definition.  If it is not on the path, the
# equivalent definitions below are used instead, and everything in this file
# (and in the timescales notebook) works standalone.
# ---------------------------------------------------------------------------
try:
    from model import Params, E_star, jacobian, g_hopf          # noqa: F401
    _USING_EXTERNAL_MODEL = True

except ImportError:
    _USING_EXTERNAL_MODEL = False

    @dataclass(frozen=True)
    class Params:
        """Parameters of the deterministic role-reversal model."""
        r:  float = 0.4     # prey intrinsic growth rate
        a:  float = 0.01    # prey inverse carrying capacity
        b:  float = 0.4     # prey consumption by adult predators
        k:  float = 0.3     # conversion of consumed prey into juveniles
        D:  float = 0.4     # maturation rate, juvenile -> adult
        v1: float = 0.2     # juvenile predator mortality
        v2: float = 0.2     # adult predator mortality
        s:  float = 0.2     # prey gain from killing juveniles
        g:  float = 0.45    # prey attack rate on juveniles

        def with_(self, **kw):
            """Return a copy with the named fields replaced."""
            return replace(self, **kw)

    def E_star(p: Params):
        """Coexistence equilibrium (x*, y1*, y2*)."""
        den = p.k*p.D - p.g*p.v2
        x = p.v2*(p.D + p.v1)/den
        f = (p.a*x - p.r)/(p.s*p.v2 - p.b*p.D)
        return np.array([x, p.v2*f, p.D*f])

    def jacobian(E, p: Params):
        """Jacobian of the deterministic model at the state E = (x, y1, y2)."""
        x, y1, y2 = E
        return np.array([
            [p.r - 2*p.a*x + p.s*y1 - p.b*y2, p.s*x,                 -p.b*x],
            [p.k*y2 - p.g*y1,                 -(p.g*x + p.D + p.v1),   p.k*x],
            [0.0,                              p.D,                   -p.v2],
        ])

    def g_star(p: Params):
        """Transcritical value: E* collides with the predator-free equilibrium."""
        return (p.r*p.k*p.D - p.a*p.v2*(p.D + p.v1))/(p.r*p.v2)

    def g_hopf(p: Params):
        """
        Supercritical Hopf value g_1, i.e. the root of B_1 B_2 - B_3 with
        B_2 > 0, where lambda^3 + B_1 lambda^2 + B_2 lambda + B_3 is the
        characteristic polynomial of J_0 at E*.
        """
        from scipy.optimize import brentq

        def coeffs(g):
            pp = p.with_(g=g)
            J = jacobian(E_star(pp), pp)
            B1 = -np.trace(J)
            B3 = -np.linalg.det(J)
            B2 = sum(np.linalg.det(np.delete(np.delete(J, i, 0), i, 1))
                     for i in range(3))
            return B1, B2, B3

        def rh(g):
            B1, B2, B3 = coeffs(g)
            return B1*B2 - B3

        # admissible range: E* must exist (g < g*) and kD - g v2 > 0
        hi = min(g_star(p), p.k*p.D/p.v2) - 1e-6
        grid = np.linspace(1e-3, hi, 2001)
        vals = np.array([rh(g) for g in grid])
        sign_change = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
        for i in sign_change:
            root = brentq(rh, grid[i], grid[i+1])
            if coeffs(root)[1] > 0:            # B_2 > 0 at a genuine Hopf
                return root
        raise RuntimeError("no Hopf bifurcation found in the admissible range")


def g_min_default(p: Params) -> float:
    """Lower barrier g_min = s*k/b (Proposition 1)."""
    return p.s * p.k / p.b


@dataclass
class OUNoise:
    """
    exp-OU driver for the prey attack rate.

    Fields
    ------
    g_bar   : fixed point (median) of G; the value of G at U = 0.
    sigma   : sigma_U, the noise intensity of the *latent* OU process U.
    alpha   : mean-reversion rate of U (1/alpha = correlation time).
    g_min   : lower barrier of the support of G.  If None, s*k/b of the
              Params passed to the integrator is used.

    Note: `sigma` is the noise on U, not on G.  To fix the standard deviation
    of G itself, build the object with OUNoise.from_std_g(...).
    """
    g_bar: float
    sigma: float
    alpha: float
    g_min: Optional[float] = None

    # ---- stationary quantities of the latent process U --------------------
    @property
    def stationary_var_U(self):
        """v = sigma_U^2 / (2*alpha), the stationary variance of U."""
        return self.sigma**2 / (2.0 * self.alpha)

    @property
    def stationary_std_U(self):
        return np.sqrt(self.stationary_var_U)

    # ---- stationary quantities of the attack rate G -----------------------
    def delta(self, p: Params = None):
        gm = self._resolve_g_min(p)
        return self.g_bar - gm

    def stationary_mean_G(self, p: Params = None):
        """E[G] = g_min + Delta*exp(v/2)  (exceeds g_bar by O(v))."""
        v = self.stationary_var_U
        return self._resolve_g_min(p) + self.delta(p) * np.exp(0.5 * v)

    def stationary_std_G(self, p: Params = None):
        """sqrt(Var(G)) = Delta*sqrt(exp(v)*(exp(v)-1))."""
        v = self.stationary_var_U
        return self.delta(p) * np.sqrt(np.exp(v) * (np.exp(v) - 1.0))

    def _resolve_g_min(self, p: Params = None):
        if self.g_min is not None:
            return self.g_min
        if p is None:
            raise ValueError("g_min is None; pass Params or set g_min explicitly.")
        return g_min_default(p)

    # ---- construction from a target stationary variance of U --------------
    @classmethod
    def from_var_u(cls, g_bar, v, alpha, g_min):
        """
        Build a driver whose latent process U has stationary variance
        v = sigma_U^2 / (2*alpha), i.e. sigma_U = sqrt(2*alpha*v).

        Holding v fixed while alpha varies keeps the driver's amplitude
        constant -- both std(U) = sqrt(v) and, since Delta is fixed,
        std(G) = Delta*sqrt(e^v (e^v - 1)) -- so that a sweep in alpha
        isolates the effect of the correlation time.
        """
        if g_bar - g_min <= 0:
            raise ValueError(f"g_bar={g_bar} must exceed g_min={g_min}.")
        return cls(g_bar=g_bar, sigma=float(np.sqrt(2.0*alpha*v)),
                   alpha=alpha, g_min=g_min)

    # ---- construction from a target std(G) --------------------------------
    @classmethod
    def from_std_g(cls, g_bar, std_g, alpha, g_min, exact=True):
        """
        Build a driver whose stationary standard deviation of G equals std_g.

        With w = (std_g/Delta)^2 the exact inversion of
        Var(G) = Delta^2 e^v (e^v - 1) is

            v = ln( (1 + sqrt(1 + 4w)) / 2 ),      sigma_U = sqrt(2*alpha*v).

        With exact=False the leading-order relation sigma_U = std_g*sqrt(2*alpha)/Delta
        is used instead; the two agree to O(v).
        """
        delta = g_bar - g_min
        if delta <= 0:
            raise ValueError(f"g_bar={g_bar} must exceed g_min={g_min}.")
        if exact:
            w = (std_g / delta) ** 2
            v = np.log(0.5 * (1.0 + np.sqrt(1.0 + 4.0 * w)))
            sigma_U = np.sqrt(2.0 * alpha * v)
        else:
            sigma_U = std_g * np.sqrt(2.0 * alpha) / delta
        return cls(g_bar=g_bar, sigma=float(sigma_U), alpha=alpha, g_min=g_min)


def simulate_exp_ou_sde(p: Params, ou: OUNoise, t_end, dt=DT_DEFAULT,
                        ic=(0.5, 0.8, 0.5), u0=0.0,
                        seed=None, store_every=1, log_state=True):
    """
    Integrate the exp-OU-augmented system.

    Parameters
    ----------
    p           : Params
    ou          : OUNoise (exp-OU driver; ou.sigma is the noise on U)
    t_end, dt   : horizon and timestep (dt defaults to 1e-3)
    ic          : (x0, y1_0, y2_0), strictly positive
    u0          : initial value of the latent driver (0 puts G at g_bar)
    seed        : RNG seed
    store_every : store every k-th step
    log_state   : integrate populations in log coordinates (default True)

    Returns
    -------
    t : (N,) array of times
    X : (N, 4) array of states (x, y1, y2, G), with G = g_min + Delta*exp(U)
    """
    g_min = ou._resolve_g_min(p)
    delta = ou.g_bar - g_min
    if delta <= 0:
        raise ValueError(f"g_bar={ou.g_bar} must exceed g_min={g_min}.")

    ic = np.asarray(ic, dtype=float)
    if np.any(ic <= 0.0):
        raise ValueError("initial populations must be strictly positive.")

    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(t_end / dt))
    n_store = n_steps // store_every + 1

    r, a, b, k, D, v1, v2, s = p.r, p.a, p.b, p.k, p.D, p.v1, p.v2, p.s
    alpha, sigma_U = ou.alpha, ou.sigma
    sqrt_dt = np.sqrt(dt)

    # ---- population right-hand side, scalar arithmetic for speed ----------
    if log_state:
        def rhs(w1, w2, w3, G):
            """d/dt of (log x, log y1, log y2)."""
            x = np.exp(w1); y1 = np.exp(w2); y2 = np.exp(w3)
            return (r - a*x + s*y1 - b*y2,
                    k*x*y2/y1 - (G*x + D + v1),
                    D*y1/y2 - v2)
    else:
        def rhs(x, y1, y2, G):
            return (x*(r - a*x + s*y1 - b*y2),
                    k*x*y2 - y1*(G*x + D + v1),
                    D*y1 - v2*y2)

    w1, w2, w3 = (np.log(ic) if log_state else ic)
    w1, w2, w3 = float(w1), float(w2), float(w3)
    U = float(u0)

    t_out = np.empty(n_store)
    X_out = np.empty((n_store, 4))
    t_out[0] = 0.0
    if log_state:
        X_out[0, :3] = (np.exp(w1), np.exp(w2), np.exp(w3))
    else:
        X_out[0, :3] = (w1, w2, w3)
    X_out[0, 3] = g_min + delta*np.exp(U)

    # Gaussian increments drawn in chunks (cheaper than one draw per step,
    # without materializing an array of length n_steps).
    CHUNK = 1_000_000
    noise = rng.standard_normal(min(CHUNK, n_steps))
    n_i = 0

    dt2 = 0.5*dt
    dt6 = dt/6.0
    one_minus = 1.0 - alpha*dt

    store_idx = 1
    for step in range(1, n_steps + 1):
        if n_i == noise.size:
            noise = rng.standard_normal(min(CHUNK, n_steps - step + 1))
            n_i = 0
        xi = noise[n_i]; n_i += 1

        # --- Euler-Maruyama for the latent OU driver ---------------------
        U_new = one_minus*U + sigma_U*sqrt_dt*xi
        U_mid = 0.5*(U + U_new)

        G_0   = g_min + delta*np.exp(U)
        G_mid = g_min + delta*np.exp(U_mid)
        G_1   = g_min + delta*np.exp(U_new)

        # --- classical RK4 for the populations ---------------------------
        a1, a2, a3 = rhs(w1, w2, w3, G_0)
        b1, b2, b3 = rhs(w1 + dt2*a1, w2 + dt2*a2, w3 + dt2*a3, G_mid)
        c1, c2, c3 = rhs(w1 + dt2*b1, w2 + dt2*b2, w3 + dt2*b3, G_mid)
        d1, d2, d3 = rhs(w1 + dt*c1,  w2 + dt*c2,  w3 + dt*c3,  G_1)

        w1 += dt6*(a1 + 2.0*b1 + 2.0*c1 + d1)
        w2 += dt6*(a2 + 2.0*b2 + 2.0*c2 + d2)
        w3 += dt6*(a3 + 2.0*b3 + 2.0*c3 + d3)
        U = U_new

        if step % store_every == 0:
            t_out[store_idx] = step*dt
            if log_state:
                X_out[store_idx, :3] = (np.exp(w1), np.exp(w2), np.exp(w3))
            else:
                X_out[store_idx, :3] = (w1, w2, w3)
            X_out[store_idx, 3] = G_1
            store_idx += 1

    return t_out[:store_idx], X_out[:store_idx]


# Backwards-compatible name: the driver is now exp-OU, not OU.
simulate_ou_sde = simulate_exp_ou_sde


def stationary_stats(p: Params, ou: OUNoise, t_end=5000.0, dt=DT_DEFAULT,
                     t_burn=1000.0, ic=(6.0, 0.6, 1.2), seed=None,
                     store_every=None):
    """
    Run a long simulation, discard burn-in, and return summary statistics.

    Returns a dict with keys t, X, mean, std, min, max.
    """
    if store_every is None:
        store_every = max(1, int(round(0.1/dt)))
    t, X = simulate_exp_ou_sde(p, ou, t_end=t_end, dt=dt, ic=ic, seed=seed,
                               store_every=store_every)
    mask = t >= t_burn
    Xs = X[mask]
    return {
        "t":    t[mask],
        "X":    Xs,
        "mean": Xs.mean(axis=0),
        "std":  Xs.std(axis=0),
        "min":  Xs.min(axis=0),
        "max":  Xs.max(axis=0),
    }
