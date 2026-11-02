"""All 2-D stochastic processes of radonika.

Every process is a 2-D process (X_t, Y_t).  A scalar parameter is used for both
coordinates; pass a length-2 list to give each coordinate its own value.

    m = rd.bm(0.5, 1.0, seed=1)
    m.plot_law(time_slice=15, n=20)
"""
from __future__ import annotations

import numpy as np
from scipy import stats

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

    # ---- plumbing
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
        """Creates NumPy's modern Generator using instance or local seed."""
        return np.random.default_rng(self.seed if seed is None else seed)

    # ---- to be provided by subclasses
    def simulate(self, n, times, start, rng):
        """Paths of shape (n, len(times), dim); times[0] must be 0 and X_0 = start."""
        raise NotImplementedError

    def marginal(self, t, start, k):
        """Frozen scipy distribution of coordinate k at time t (when known in closed form)."""
        raise NotImplementedError

    # ---- laws
    def pdf(self, t, x, y, start=None):
        """Closed-form density (pmf for lattice laws) of (X_t, Y_t) at (x, y), or None.
        
        Product rule for independent coordinates: f(x, y) = f(x) * f(y)
        """
        if not self.independent:
            return None
        s = self._start(start)
        out = 1.0
        for k, c in enumerate((x, y)):
            d = self.marginal(t, s, k)
            f = d.pmf(np.rint(c)) if isinstance(d.dist, stats.rv_discrete) else d.pdf(c)
            out = out * f
        return out


    def sample_slices(self, times, n, rng, start, fine=None):
        """Simulates up to slice times and extracts the state at those times. Exact for all built-in processes."""
        X = self.simulate(n, np.r_[0.0, np.asarray(times, float)], start, rng)
        return np.transpose(X[:, 1:, :], (1, 0, 2))
        # Transpose from (n, slices, dim) -> (slices, n, dim)

    # ---- user-friendly sampling
    def paths(self, n=20, T=1.0, steps=200, start=None, seed=None):
        """Simulate n paths on [0, T]; returns (times, X) with X of shape (n, steps+1, 2)."""
        times = np.linspace(0.0, T, steps + 1)
        return times, self.simulate(n, times, self._start(start), self._rng(seed))

    def sample(self, t, n=10_000, start=None, seed=None):
        """n independent samples of X_t, shape (n, 2) ie, at fixed time."""
        return self.sample_slices([t], n, self._rng(seed), self._start(start))[0]


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

# --------------------------------------------------------------------------- Brownian family
class bm(Process):
    """Brownian motion with drift: X_t = x0 + mu t + A W_t.   Levy, has an MGF.

    Parameters
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
        s = self._start(start)
        return _bvn_pdf(x, y, s + self.mu * t, self.Sigma * t)

