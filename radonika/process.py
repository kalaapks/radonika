"""All 2-D stochastic processes of radonika.

Every process is a 2-D process (X_t, Y_t).  A scalar parameter is used for both
coordinates; pass a length-2 list to give each coordinate its own value.

    m = rd.bm(0.5, 1.0, seed=1)
    m.plot_law(time_slice=15, n=20)
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from ._utils import hist_density
from .display import fmt, show_math

__all__ = [
    "Process", "normal", "cauchy", "exponential", "uniform", "student_t", "laplace",
    "lognormal", "iid", "bm", "bm_bridge", "poisson", "gamma_process",
    "cauchy_process", "stocks", "stocks_neutral",
]

NOT_LEVY_MSG = ("This is not a Levy process. Reshifting every time slice to remove drift "
                "using Esscher will force us to tear the process apart.")
NO_MGF_MSG = ("This is a Levy process, but it has no exponential moments: E[exp(theta*X_t)] is "
              "infinite for every theta != 0, so the Esscher transform does not exist.")


# --------------------------------------------------------------------------- helpers
def _pair(v, name="parameter"):
    a = np.asarray(v, float)
    if a.ndim == 0:
        return np.repeat(a, 2)
    if a.shape == (2,):
        return a.copy()
    raise ValueError(f"{name} must be a scalar (used for both coordinates) or a length-2 sequence")


def _vol(sigma, rho):
    if not -1 < rho < 1:
        raise ValueError("rho must lie strictly between -1 and 1")
    L = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho ** 2)]])
    return sigma[:, None] * L  # A = diag(sigma) L, covariance Sigma = A A^T


def _bvn_pdf(x, y, mean, cov):
    dx, dy = x - mean[0], y - mean[1]
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
    q = (cov[1, 1] * dx * dx - 2 * cov[0, 1] * dx * dy + cov[0, 0] * dy * dy) / det
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))


def _gauss_paths(n, times, drift, A, rng):
    """Cumulative Gaussian increments (n, len(times), 2): Y_t = drift*t + A W_t, Y_0 = 0."""
    dt = np.diff(times)
    z = rng.standard_normal((n, len(dt), 2))
    inc = np.sqrt(dt)[None, :, None] * (z @ A.T) + drift[None, None, :] * dt[None, :, None]
    return np.concatenate([np.zeros((n, 1, 2)), np.cumsum(inc, axis=1)], axis=1)


# --------------------------------------------------------------------------- base class
class Process:
    """Base class: every process knows how to simulate itself and (where possible) its law."""

    name = "process"
    dim = 2
    is_levy = False          # genuine Levy process (independent, stationary increments)
    has_mgf = False          # exponential moments exist
    esscher_space = None     # None | "raw" | "log"  (space in which Esscher tilting acts)
    horizon = None           # finite time horizon (bridge) or None
    default_start = (0.0, 0.0)
    measures: tuple = ()     # alternative measures offered by the viewer's toggle
    independent = True       # coordinates independent -> pdf is a product of marginals
    _desc = "process"

    def __init__(self, seed=None):
        self.seed = seed

    def __repr__(self):
        return self._desc

    def _start(self, start):
        """Validates or assigns default starting vector (0.0, 0.0)."""
        s = self.default_start if start is None else start
        s = np.asarray(s, float)
        if s.ndim == 0:
            s = np.repeat(s, self.dim)
        if s.shape != (self.dim,):
            raise ValueError(f"start must have {self.dim} coordinates")
        return s

    def _rng(self, seed=None):
        return np.random.default_rng(self.seed if seed is None else seed)

    #---------------------------------------------- to be provided by subclasses
    def simulate(self, n, times, start, rng):
        """Paths of shape (n, len(times), dim); times[0] must be 0 and X_0 = start."""
        raise NotImplementedError

    def marginal(self, t, start, k):
        """Frozen scipy distribution of coordinate k at time t (when known in closed form)."""
        raise NotImplementedError

    #--------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------- laws
    def pdf(self, t, x, y, start=None):
        """Closed-form density (pmf for lattice laws) of (X_t, Y_t) at (x, y), or None.
        
        Product rule for independent coordinates: f(x, y) = f(x) * f(y)
        """
        if t <= 0 or not self.independent:      # t = 0 is a point mass at `start`: no density
            return None
        s = self._start(start)
        out = 1.0
        for k, c in enumerate((x, y)):
            d = self.marginal(t, s, k)
            f = d.pmf(np.rint(c)) if isinstance(d.dist, stats.rv_discrete) else d.pdf(c)
            out = out * f
        return out

    def center(self, t, start=None):
        """Mean of X_t (median if the mean does not exist, e.g. Cauchy)."""
        s = self._start(start)
        if t <= 0:
            return s.copy()
        out = []
        for k in range(self.dim):
            d = self.marginal(t, s, k)
            m = d.mean()
            out.append(m if np.isfinite(m) else d.median())
        return np.array(out, float)
    
    #the next 4 line commented because i want the time slider to start at 0 without causing an issue with the pdf
    
    # def sample_slices(self, times, n, rng, start, fine=None):
    #     """Simulates up to slice times and extracts the state at those times. Exact for all built-in processes."""
    #     X = self.simulate(n, np.r_[0.0, np.asarray(times, float)], start, rng)
    #     return np.transpose(X[:, 1:, :], (1, 0, 2))

    def sample_slices(self, times, n, rng, start, fine=None):
        """State at each slice time, shape (slices, n, dim).  A slice at t = 0 is exactly `start`."""
        t_arr = np.asarray(times, float)
        pos = t_arr > 0
        out = np.empty((len(t_arr), n, len(start)))
        out[~pos] = start                                   # X_0 = start (no zero-length step)
        if pos.any():
            X = self.simulate(n, np.r_[0.0, t_arr[pos]], start, rng)
            out[pos] = np.transpose(X[:, 1:, :], (1, 0, 2))  # (n, slices, dim) -> (slices, n, dim)
        return out

    def paths(self, n=20, T=1.0, steps=200, start=None, seed=None):
        """Simulate n paths on [0, T]; returns (times, X) with X of shape (n, steps+1, 2)."""
        times = np.linspace(0.0, T, steps + 1)
        return times, self.simulate(n, times, self._start(start), self._rng(seed))

    def sample(self, t, n=10000, start=None, seed=None):
        """n independent samples of X_t, shape (n, 2) ie, at fixed time."""
        return self.sample_slices([t], n, self._rng(seed), self._start(start))[0]

    # ---- Esscher / measure machinery
    def cumulant(self, theta):
        raise NotImplementedError

    def levy_state(self, X, start):
        return X - start

    def esscher(self, theta):
        raise NotImplementedError

    def theta_domain(self):
        return -np.inf, np.inf

    def theta_star(self):
        """Tilt (applied to both coordinates) that removes the drift, or None."""
        return None

    def at(self, theta=0.0, measure="P"):
        """Closed-form variant of the process under an Esscher tilt / alternative measure."""
        if theta == 0 and measure == "P":
            return self
        if self.esscher_space is None or measure != "P":
            raise NotImplementedError(f"{self!r} does not support tilting / measure change")
        return self.esscher(np.array([theta, theta], float))

    def weights(self, t, X, start, theta=0.0, measure="P"):
        """Radon-Nikodym weights dQ/dP at time t evaluated on P-samples X (N, dim)."""
        w = np.ones(len(X))
        if theta != 0 and self.esscher_space:
            th = np.array([theta, theta], float)
            w = w * np.exp(self.levy_state(X, start) @ th - t * self.cumulant(th))
        return w

    # ---- visual entry points (lazy imports keep the package light)
    def plot_law(self, start=None, time=None, time_slice=None, n=20, seed=None, **kw):
        """Interactive paths (right) + density surface (left) in the browser.

        start       : starting point (x0, y0); default origin (1, 1 for stock prices)
        time        : a single time (no slider) or a list of slice times
        time_slice  : integer L -> slider over [T/L, 2T/L, ..., T]
        n           : number of displayed paths
        Other options: T, width, height, controls, autoplay, interval, loop, n_density,
        grid, method ('closed' | 'reweight'), port, open_browser, block.
        """
        from .viz_dash import plot_law

        return plot_law(self, start=start, time=time, time_slice=time_slice, n=n, seed=seed, **kw)

    def tilt(self, start=None, time=None, time_slice=None, n=20, theta_range=(-2.0, 2.0),
             seed=None, **kw):
        """plot_law plus a horizontal Esscher-theta slider (Levy processes with an MGF only)."""
        if self.is_levy and not self.has_mgf:
            print(NO_MGF_MSG)
            return None
        if self.esscher_space is None:
            print(NOT_LEVY_MSG)
            return None
        if self.esscher_space == "log":
            print("Note: S_t is not Levy but log S_t is, so the Esscher tilt acts on log S_t.")
        from .viz_dash import plot_law

        return plot_law(self, start=start, time=time, time_slice=time_slice, n=n, seed=seed,
                        tilt=True, theta_range=theta_range, **kw)

    def analyze(self, t=4, start=None, times=None, T=1.0, n=400, n_paths=60, seed=None,
                save=None, show=True):
        """Static matplotlib report: t slices (<= 6) on top, contour evolution below."""
        from .viz_static import analyze

        return analyze(self, t=t, start=start, times=times, T=T, n=n, n_paths=n_paths,
                       seed=seed, save=save, show=show)

    def rare(self, K, T=1.0, start=None, n_max=100_000, seed=None, show=True):
        """Importance sampling of P(X^1_T > K) with an Esscher tilt vs naive Monte Carlo."""
        from .measure import rare_event
        from .viz_static import plot_rare

        res = rare_event(self, K, T=T, start=start, n_max=n_max, seed=seed)
        plot_rare(res, show=show)
        return res


# --------------------------------------------------------------------------- iid processes
class _IID(Process):
    """X_t independent across time; coordinates independent scipy distributions."""

    def __init__(self, frozen, seed=None, desc="iid"):
        super().__init__(seed)
        self.frozen = list(frozen)
        self._desc = desc

    def simulate(self, n, times, start, rng):
        out = np.empty((n, len(times), 2))
        out[:, 0, :] = start
        for k in range(2):
            out[:, 1:, k] = self.frozen[k].rvs(size=(n, len(times) - 1), random_state=rng)
        return out

    def marginal(self, t, start, k):
        return self.frozen[k]

    def logpdf(self, X):
        return sum(self.frozen[k].logpdf(X[..., k]) for k in range(2))


def iid(dist, seed=None):
    """iid(scipy.stats.gamma(2)) or iid([frozen_x, frozen_y]): any scipy distribution."""
    pair = list(dist) if isinstance(dist, (list, tuple)) else [dist, dist]
    return _IID(pair, seed, desc=f"iid({pair[0].dist.name})")


class normal(_IID):
    """iid Normal(mean, sigma) noise at every time step (sigma = standard deviation)."""

    def __init__(self, mean=0.0, sigma=1.0, seed=None):
        m, s = _pair(mean, "mean"), _pair(sigma, "sigma")
        super().__init__([stats.norm(m[k], s[k]) for k in range(2)], seed,
                         f"normal(mean={fmt(m)}, sigma={fmt(s)})")


class cauchy(_IID):
    """iid Cauchy(loc, scale) noise (no mean, no variance, heavy tails)."""

    def __init__(self, loc=0.0, scale=1.0, seed=None):
        l, s = _pair(loc, "loc"), _pair(scale, "scale")
        super().__init__([stats.cauchy(l[k], s[k]) for k in range(2)], seed,
                         f"cauchy(loc={fmt(l)}, scale={fmt(s)})")


class exponential(_IID):
    """iid Exponential(rate) noise."""

    def __init__(self, rate=1.0, seed=None):
        r = _pair(rate, "rate")
        super().__init__([stats.expon(scale=1 / r[k]) for k in range(2)], seed,
                         f"exponential(rate={fmt(r)})")


class uniform(_IID):
    """iid Uniform(low, high) noise."""

    def __init__(self, low=0.0, high=1.0, seed=None):
        a, b = _pair(low, "low"), _pair(high, "high")
        super().__init__([stats.uniform(a[k], b[k] - a[k]) for k in range(2)], seed,
                         f"uniform(low={fmt(a)}, high={fmt(b)})")


class student_t(_IID):
    """iid Student-t(df, loc, scale) noise."""

    def __init__(self, df=3.0, loc=0.0, scale=1.0, seed=None):
        d, l, s = _pair(df, "df"), _pair(loc, "loc"), _pair(scale, "scale")
        super().__init__([stats.t(d[k], l[k], s[k]) for k in range(2)], seed,
                         f"student_t(df={fmt(d)}, loc={fmt(l)}, scale={fmt(s)})")


class laplace(_IID):
    """iid Laplace(loc, scale) noise."""

    def __init__(self, loc=0.0, scale=1.0, seed=None):
        l, s = _pair(loc, "loc"), _pair(scale, "scale")
        super().__init__([stats.laplace(l[k], s[k]) for k in range(2)], seed,
                         f"laplace(loc={fmt(l)}, scale={fmt(s)})")


class lognormal(_IID):
    """iid LogNormal noise: exp(Normal(mean, sigma))."""

    def __init__(self, mean=0.0, sigma=1.0, seed=None):
        m, s = _pair(mean, "mean"), _pair(sigma, "sigma")
        super().__init__([stats.lognorm(s[k], scale=np.exp(m[k])) for k in range(2)], seed,
                         f"lognormal(mean={fmt(m)}, sigma={fmt(s)})")


# --------------------------------------------------------------------------- Brownian family
class bm(Process):
    """Brownian motion with drift: X_t = x0 + mu t + A W_t.   Levy, has an MGF.

    Parameters:
    ----------
    mu, sigma : scalar or length-2 (drift and volatility per coordinate)
    rho       : correlation between the two coordinates
    """

    is_levy = True
    has_mgf = True
    esscher_space = "raw"

    def __init__(self, mu=0.0, sigma=1.0, seed=None, rho=0.0):
        super().__init__(seed)
        self.mu, self.sigma, self.rho = _pair(mu, "mu"), _pair(sigma, "sigma"), float(rho)
        self.A = _vol(self.sigma, self.rho)
        self.Sigma = self.A @ self.A.T
        self.independent = self.rho == 0.0
        self._desc = f"bm(mu={fmt(self.mu)}, sigma={fmt(self.sigma)}" + (
            f", rho={rho:g})" if rho else ")")

    def _gauss_params(self):
        return self.mu, self.A

    def simulate(self, n, times, start, rng):
        return start + _gauss_paths(n, times, self.mu, self.A, rng)

    def marginal(self, t, start, k):
        return stats.norm(start[k] + self.mu[k] * t, np.sqrt(self.Sigma[k, k] * t))

    def pdf(self, t, x, y, start=None):
        if t <= 0:
            return None
        s = self._start(start)
        return _bvn_pdf(x, y, s + self.mu * t, self.Sigma * t)

    def cumulant(self, theta):
        return float(self.mu @ theta + 0.5 * theta @ self.Sigma @ theta)

    def esscher(self, theta):
        return bm(self.mu + self.Sigma @ np.asarray(theta, float), self.sigma, self.seed, self.rho)

    def theta_star(self):
        s1 = self.Sigma @ np.ones(2)
        return float(-(s1 @ self.mu) / (s1 @ s1))


class bm_bridge(Process):
    """Brownian bridge from the start point to `end` at time T.  Not Levy."""

    default_start = (0.0, 0.0)

    def __init__(self, sigma=1.0, end=(0.0, 0.0), T=1.0, seed=None):
        super().__init__(seed)
        self.sigma, self.end, self.T = _pair(sigma, "sigma"), _pair(end, "end"), float(T)
        self.horizon = self.T
        self._desc = f"bm_bridge(sigma={fmt(self.sigma)}, end={fmt(self.end)}, T={self.T:g})"

    def simulate(self, n, times, start, rng):
        if times[-1] > self.T + 1e-12:
            raise ValueError(f"a bridge only lives on [0, {self.T:g}]")
        X = np.empty((n, len(times), 2))
        X[:, 0, :] = cur = np.broadcast_to(start, (n, 2)).copy()
        for i in range(1, len(times)):
            t0, t1 = times[i - 1], times[i]
            rem, dt = self.T - t0, t1 - t0
            if rem <= 1e-12:
                cur = np.broadcast_to(self.end, (n, 2)).copy()
            else:
                mean = cur + (self.end - cur) * dt / rem
                var = self.sigma ** 2 * dt * max(self.T - t1, 0.0) / rem
                cur = mean + np.sqrt(var) * rng.standard_normal((n, 2))
            X[:, i, :] = cur
        return X

    def marginal(self, t, start, k):
        mean = start[k] + (t / self.T) * (self.end[k] - start[k])
        var = self.sigma[k] ** 2 * t * (self.T - t) / self.T
        return stats.norm(mean, max(np.sqrt(max(var, 0.0)), 1e-3 * self.sigma[k]))


# --------------------------------------------------------------------------- jump / Levy
class poisson(Process):
    """Poisson process with intensity `rate` per coordinate (subordinator, Levy)."""

    is_levy = True
    has_mgf = True
    esscher_space = "raw"

    def __init__(self, rate=1.0, seed=None):
        super().__init__(seed)
        self.rate = _pair(rate, "rate")
        self._desc = f"poisson(rate={fmt(self.rate)})"

    def simulate(self, n, times, start, rng):
        dt = np.diff(times)
        inc = rng.poisson(self.rate[None, None, :] * dt[None, :, None], size=(n, len(dt), 2))
        return start + np.concatenate([np.zeros((n, 1, 2)), np.cumsum(inc, axis=1)], axis=1)

    def marginal(self, t, start, k):
        return stats.poisson(self.rate[k] * t, loc=start[k])

    def cumulant(self, theta):
        return float(np.sum(self.rate * (np.exp(theta) - 1)))

    def esscher(self, theta):
        return poisson(self.rate * np.exp(np.asarray(theta, float)), self.seed)


class gamma_process(Process):
    """Gamma process: X_t ~ Gamma(shape a t, rate b)  (subordinator, Levy)."""

    is_levy = True
    has_mgf = True
    esscher_space = "raw"

    def __init__(self, a=1.0, b=1.0, seed=None):
        super().__init__(seed)
        self.a, self.b = _pair(a, "a"), _pair(b, "b")
        self._desc = f"gamma_process(a={fmt(self.a)}, b={fmt(self.b)})"

    def simulate(self, n, times, start, rng):
        dt = np.diff(times)
        inc = rng.gamma(self.a[None, None, :] * dt[None, :, None], 1 / self.b, size=(n, len(dt), 2))
        return start + np.concatenate([np.zeros((n, 1, 2)), np.cumsum(inc, axis=1)], axis=1)

    def marginal(self, t, start, k):
        return stats.gamma(self.a[k] * t, loc=start[k], scale=1 / self.b[k])

    def theta_domain(self):
        return -np.inf, 0.95 * float(self.b.min())

    def cumulant(self, theta):
        return float(-np.sum(self.a * np.log(1 - np.asarray(theta) / self.b)))

    def esscher(self, theta):
        th = np.asarray(theta, float)
        if np.any(th >= self.b):
            raise ValueError("theta must stay below the rate b for the Gamma MGF to exist")
        return gamma_process(self.a, self.b - th, self.seed)


class cauchy_process(Process):
    """Cauchy process: X_t ~ Cauchy(loc t, scale t).  Levy but with NO exponential moments."""

    is_levy = True
    has_mgf = False

    def __init__(self, scale=1.0, loc=0.0, seed=None):
        super().__init__(seed)
        self.scale, self.loc = _pair(scale, "scale"), _pair(loc, "loc")
        self._desc = f"cauchy_process(scale={fmt(self.scale)}, loc={fmt(self.loc)})"

    def simulate(self, n, times, start, rng):
        dt = np.diff(times)
        inc = rng.standard_cauchy((n, len(dt), 2)) * self.scale * dt[None, :, None] \
            + self.loc * dt[None, :, None]
        return start + np.concatenate([np.zeros((n, 1, 2)), np.cumsum(inc, axis=1)], axis=1)

    def marginal(self, t, start, k):
        return stats.cauchy(start[k] + self.loc[k] * t, self.scale[k] * t)


# --------------------------------------------------------------------------- stocks
class stocks(Process):
    """Discounted 2-D stock prices under the real-world measure P.

        d S~_i = ((mu_i - r) dt + sum_j A_ij dB_j) S~_i

    Parameters
    ----------
    mu, sigma : drift and volatility (scalar or length-2)
    r         : risk-free rate
    rho       : correlation of the two Brownian motions
    The start (initial prices) defaults to (1, 1).

    The viewer has a measure toggle: P, Q (risk neutral), Q1 / Q2 (numeraire = stock 1 / 2).
    """

    esscher_space = "log"      # S is not Levy, log S is
    default_start = (1.0, 1.0)
    measures = ("P", "Q", "Q1", "Q2")
    _neutral = False

    def __init__(self, mu, sigma, r, seed=None, rho=0.0, verbose=True):
        super().__init__(seed)
        self.mu, self.sigma, self.r, self.rho = _pair(mu, "mu"), _pair(sigma, "sigma"), float(r), float(rho)
        self.A = _vol(self.sigma, self.rho)
        self.Sigma = self.A @ self.A.T
        self.independent = self.rho == 0.0
        self._desc = (f"{type(self).__name__}(mu={fmt(self.mu)}, sigma={fmt(self.sigma)}, r={self.r:g}"
                      + (f", rho={rho:g})" if rho else ")"))
        if verbose:
            self._announce()

    # -- drifts of log S~ (the Levy part)
    @property
    def log_drift(self):
        return self.mu - self.r - 0.5 * np.diag(self.Sigma)

    def _drift_under(self, measure):
        half = 0.5 * np.diag(self.Sigma)
        if measure == "P":
            return self.log_drift
        if measure == "Q":
            return -half
        if measure in ("Q1", "Q2"):
            return self.Sigma[:, int(measure[1]) - 1] - half
        raise ValueError(f"unknown measure {measure!r}")

    def _gauss_params(self):
        return self.log_drift, self.A

    def _announce(self):
        mu, sg, r = fmt(self.mu), fmt(self.sigma), f"{self.r:g}"
        tex = [
            (rf"d\tilde{{S}}_t = \left(({mu}-{r})\,dt + {sg}\,dB_t\right)\tilde{{S}}_t",
             f"dS̃_t = (({mu} − {r}) dt + {sg} dB_t) S̃_t"),
            (rf"\tilde{{S}}_t = \tilde{{S}}_0 \exp\left(\left({mu}-{r}-\tfrac{{1}}{{2}}({sg})^2\right)t + {sg}\,B_t\right)",
             f"S̃_t = S̃_0 exp( ({mu} − {r} − ½·{sg}²) t + {sg} B_t )"),
        ]
        show_math("Discounted stock price SDE:", tex)

    # -- law
    def simulate(self, n, times, start, rng):
        return start * np.exp(_gauss_paths(n, times, self.log_drift, self.A, rng))

    def marginal(self, t, start, k):
        return stats.lognorm(np.sqrt(self.Sigma[k, k] * t), scale=start[k] * np.exp(self.log_drift[k] * t))

    def pdf(self, t, x, y, start=None):
        if t <= 0:
            return None
        s = self._start(start)
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = (x > 0) & (y > 0)
        lx, ly = np.log(np.where(ok, x, 1.0) / s[0]), np.log(np.where(ok, y, 1.0) / s[1])
        p = _bvn_pdf(lx, ly, self.log_drift * t, self.Sigma * t) / (np.where(ok, x, 1.0) * np.where(ok, y, 1.0))
        return np.where(ok, p, 0.0)

    # -- Esscher on log S, measure changes
    def levy_state(self, X, start):
        return np.log(X / start)

    def cumulant(self, theta):
        m = self.log_drift
        return float(m @ theta + 0.5 * theta @ self.Sigma @ theta)

    def _clone(self, log_drift):
        mu = log_drift + self.r + 0.5 * np.diag(self.Sigma)
        return stocks(mu, self.sigma, self.r, self.seed, self.rho, verbose=False)

    def esscher(self, theta):
        return self._clone(self.log_drift + self.Sigma @ np.asarray(theta, float))

    def theta_star(self):
        s1 = self.Sigma @ np.ones(2)
        return float(-(s1 @ (self.mu - self.r)) / (s1 @ s1))

    def at(self, theta=0.0, measure="P"):
        if theta == 0 and measure == "P":
            return self
        return self._clone(self._drift_under(measure) + self.Sigma @ np.array([theta, theta], float))

    def weights(self, t, X, start, theta=0.0, measure="P"):
        """dQ/dP at time t: Z = exp(-W.th - |th|^2 t / 2), th = A^-1 (drift_P - drift_Q)."""
        if measure == "P" and theta == 0:
            return np.ones(len(X))
        m_from = self.log_drift
        m_to = self._drift_under(measure) + self.Sigma @ np.array([theta, theta], float)
        Y = np.log(X / start)
        W = np.linalg.solve(self.A, (Y - m_from * t).T).T
        th = np.linalg.solve(self.A, m_from - m_to)
        return np.exp(-W @ th - 0.5 * th @ th * t)

    # -- numeraire demo: Margrabe exchange option
    def exchange_option(self, T=1.0, start=None, n=400_000, seed=None):
        """Price of the option to swap stock 2 for stock 1 (payoff (S1-S2)^+, discounted units).

        Compares Margrabe's formula with Monte Carlo under Q (money-market numeraire)
        and under Q1 (numeraire = stock 1, the change of numeraire Z = S1_T / S1_0).
        """
        s = self._start(start)
        rng = self._rng(seed)
        S, half = self.Sigma, 0.5 * np.diag(self.Sigma)
        v = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
        d1 = (np.log(s[0] / s[1]) + 0.5 * v * v * T) / (v * np.sqrt(T))
        closed = s[0] * stats.norm.cdf(d1) - s[1] * stats.norm.cdf(d1 - v * np.sqrt(T))
        z = rng.standard_normal((n, 2)) @ self.A.T * np.sqrt(T)
        XQ = s * np.exp(-half * T + z)
        mcQ = np.maximum(XQ[:, 0] - XQ[:, 1], 0)
        XQ1 = s * np.exp((S[:, 0] - half) * T + z)
        mcQ1 = s[0] * np.maximum(1 - XQ1[:, 1] / XQ1[:, 0], 0)
        out = {"margrabe": closed,
               "mc_Q": (mcQ.mean(), mcQ.std() / np.sqrt(n)),
               "mc_Q1_numeraire": (mcQ1.mean(), mcQ1.std() / np.sqrt(n))}
        print(f"Margrabe closed form : {closed:.6f}")
        print(f"Monte Carlo under Q  : {out['mc_Q'][0]:.6f} +- {out['mc_Q'][1]:.6f}")
        print(f"Monte Carlo under Q1 : {out['mc_Q1_numeraire'][0]:.6f} +- {out['mc_Q1_numeraire'][1]:.6f}")
        return out


class stocks_neutral(stocks):
    """Discounted stock prices under the risk-neutral measure Q (drift removed).

    d S~_i = sigma S~_i dB*_i with B*_t = B_t + ((mu - r)/sigma) t.
    """

    measures = ()
    _neutral = True

    def __init__(self, mu, sigma, r, seed=None, rho=0.0, verbose=True):
        self.mu_real = _pair(mu, "mu")
        super().__init__(r, sigma, r, seed, rho, verbose)   # internal drift = r  -> no drift
        self.mu_P = self.mu_real
        self._desc = (f"stocks_neutral(mu={fmt(self.mu_real)}, sigma={fmt(self.sigma)}, r={self.r:g}"
                      + (f", rho={rho:g})" if rho else ")"))

    def _announce(self):
        mu, sg, r = fmt(self.mu_real), fmt(self.sigma), f"{self.r:g}"
        tex = [
            (rf"B^*_t = B_t + \frac{{{mu}-{r}}}{{{sg}}}\,t",
             f"B*_t = B_t + (({mu} − {r}) / {sg}) t"),
            (rf"d\tilde{{S}}_t = {sg}\,dB^*_t\,\tilde{{S}}_t", f"dS̃_t = {sg} dB*_t S̃_t"),
            (rf"\tilde{{S}}_t = \tilde{{S}}_0 \exp\left(-\tfrac{{1}}{{2}}({sg})^2 t + {sg}\,B^*_t\right)",
             f"S̃_t = S̃_0 exp( −½·{sg}² t + {sg} B*_t )"),
        ]
        show_math("Discounted stock price SDE with Brownian fluctuation B^*_t = B_t + ((μ-r)/ σ) t:", tex)

    def _clone(self, log_drift):
        mu = log_drift + self.r + 0.5 * np.diag(self.Sigma)
        return stocks(mu, self.sigma, self.r, self.seed, self.rho, verbose=False)
