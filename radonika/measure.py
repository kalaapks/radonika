"""Radon-Nikodym machinery: likelihood-ratio processes and rare-event importance sampling."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ._utils import NotComparableError, SingularMeasureError, ess
from .process import (Process, _IID, bm, cauchy_process, gamma_process, poisson, stocks)

__all__ = ["lr", "LikelihoodRatio", "rare_event", "SingularMeasureError", "NotComparableError", "ess"]


def _default_fine(times, steps=240):
    Tm = float(np.max(times))
    g = np.concatenate([np.linspace(0, Tm, steps + 1), np.asarray(times, float)])
    return np.unique(g)


class LikelihoodRatio(Process):
    """Z_t = dQ/dP on F_t, simulated under P.  A *1-D* process (the value of Z_t).

    Exact (path-level) where known: drift change of bm / stocks (Girsanov), products of
    iid density ratios, Poisson and Gamma processes.  Otherwise the *marginal* ratio
    q_t(X_t)/p_t(X_t) = E^P[Z_t | X_t] is used and labelled as such.
    """

    dim = 1
    default_start = (0.0,)
    independent = False

    def __init__(self, p: Process, q: Process, seed=None):
        super().__init__(seed if seed is not None else p.seed)
        self.p, self.q = p, q
        self.kind = self._classify(p, q)
        self.exact = self.kind != "marginal"
        self.as_log = False
        self.horizon = p.horizon
        self._desc = f"lr(dQ/dP; P={p!r}, Q={q!r}, {self.kind})"

    # ------------------------------------------------------------ classification
    @staticmethod
    def _classify(p, q):
        if isinstance(p, _IID) and isinstance(q, _IID):
            for k in range(2):
                ap, bp = p.frozen[k].support()
                aq, bq = q.frozen[k].support()
                if aq < ap - 1e-12 or bq > bp + 1e-12:
                    raise SingularMeasureError(
                        "Q puts mass where P has none (support of Q is not inside support of P), "
                        "so Q is not absolutely continuous w.r.t. P and dQ/dP does not exist. "
                        "Try lr(Q, P) in the other direction.")
            return "iid"
        if type(p) is type(q) or (isinstance(p, stocks) and isinstance(q, stocks)):
            if isinstance(p, bm) or isinstance(p, stocks):
                if not (np.allclose(p.A, q.A)):
                    raise SingularMeasureError(
                        "The two diffusions have different volatility. Girsanov can change the "
                        "DRIFT of a Brownian motion but never its VOLATILITY: the realised quadratic "
                        "variation converges to sigma^2 t almost surely, so the two laws live on "
                        "disjoint sets of paths (mutually singular) and no Radon-Nikodym derivative exists.")
                return "girsanov"
            if isinstance(p, poisson):
                return "poisson"
            if isinstance(p, gamma_process):
                if not np.allclose(p.a, q.a):
                    raise SingularMeasureError(
                        "Gamma processes with different shape rates a have mutually singular laws "
                        "(Sato's theorem: the Levy measures differ in their small-jump behaviour).")
                return "gamma"
            if isinstance(p, cauchy_process):
                if np.allclose(p.scale, q.scale) and np.allclose(p.loc, q.loc):
                    return "trivial"
                raise SingularMeasureError(
                    "Cauchy processes with different scale or drift are mutually singular "
                    "(infinitely many small jumps pin down the Levy measure; Sato's theorem).")
        try:
            tp = p.pdf(0.5, np.array([0.1]), np.array([0.1]), None)
            tq = q.pdf(0.5, np.array([0.1]), np.array([0.1]), None)
        except Exception:
            tp = tq = None
        if tp is not None and tq is not None:
            return "marginal"
        raise NotComparableError("These two processes have no closed-form comparison; "
                                 "pick processes of the same family.")

    # ------------------------------------------------------------ the process Z
    def _logZ(self, X, times, start):
        n, T = X.shape[0], len(times)
        t = times[None, :]
        k = self.kind
        if k == "trivial":
            return np.zeros((n, T))
        if k == "iid":
            lr_ = self.q.logpdf(X[:, 1:, :]) - self.p.logpdf(X[:, 1:, :])
            return np.concatenate([np.zeros((n, 1)), np.cumsum(lr_, axis=1)], axis=1)
        if k == "girsanov":
            mp, A = self.p._gauss_params()
            mq, _ = self.q._gauss_params()
            Y = self.p.levy_state(X.reshape(-1, 2), start).reshape(n, T, 2)
            W = np.linalg.solve(A, (Y - mp * times[None, :, None]).reshape(-1, 2).T).T.reshape(n, T, 2)
            th = np.linalg.solve(A, mp - mq)
            return -(W @ th) - 0.5 * (th @ th) * t
        N = X - start
        if k == "poisson":
            return np.sum(N * np.log(self.q.rate / self.p.rate), axis=2) - np.sum(self.q.rate - self.p.rate) * t
        if k == "gamma":
            return (np.sum(self.p.a * np.log(self.q.b / self.p.b)) * t
                    - np.sum(N * (self.q.b - self.p.b), axis=2))
        # marginal ratio
        out = np.zeros((n, T))
        with np.errstate(all="ignore"):
            for i in range(1, T):
                pp = self.p.pdf(times[i], X[:, i, 0], X[:, i, 1], start)
                qq = self.q.pdf(times[i], X[:, i, 0], X[:, i, 1], start)
                out[:, i] = np.log(qq) - np.log(pp)
        return out

    def simulate(self, n, times, start, rng):
        X = self.p.simulate(n, times, start, rng)
        L = self._logZ(X, np.asarray(times, float), start)
        return (L if self.as_log else np.exp(L))[..., None]

    def sample_slices(self, times, n, rng, start, fine=None):
        fine = _default_fine(times) if fine is None else fine
        idx = np.searchsorted(fine, times)
        n = min(n, 8000)
        Z = self.simulate(n, fine, start, rng)[:, :, 0]
        return Z[:, idx].T[..., None]

    def marginal(self, t, start, k):
        raise NotImplementedError

    def pdf(self, t, x, y, start=None):
        return None

    def center(self, t, start=None):
        return np.array([np.nan if self.as_log else 1.0])

    def _start(self, start):
        return self.p._start(start)

    # ------------------------------------------------------------ user-facing extras
    def plot_law(self, start=None, time=None, time_slice=None, n=20, seed=None, log=None, **kw):
        """Same viewer as other processes (1-D: paths on the right, density of Z_t on the left).
        log=True shows log Z_t (default for iid products, where Z collapses towards 0 while
        E[Z] = 1 is carried by rare paths)."""
        self.as_log = (self.kind == "iid") if log is None else bool(log)
        return super().plot_law(start=start, time=time, time_slice=time_slice, n=n, seed=seed, **kw)

    def tilt(self, *a, **k):
        print("tilt is defined for the underlying processes, not for a likelihood ratio.")

    def analyze(self, *a, **k):
        print("analyze is defined for the underlying 2-D processes; use z.plot_law() or z.summary().")

    def summary(self, t=1.0, n=8000, start=None, seed=None):
        """Print E[Z], ESS and KL-rate diagnostics at time t (martingale + E log Z = -t KL)."""
        rng = self._rng(seed)
        fine = _default_fine([t])
        keep, self.as_log = self.as_log, False
        Z = self.sample_slices([t], n, rng, self._start(start), fine)[0, :, 0]
        self.as_log = keep
        kl = -np.mean(np.log(np.maximum(Z, 1e-300))) / t
        out = {"t": t, "E[Z]": Z.mean(), "ESS": ess(Z), "n": len(Z), "KL_rate_est": kl}
        if self.kind == "girsanov":
            mp, A = self.p._gauss_params()
            mq, _ = self.q._gauss_params()
            th = np.linalg.solve(A, mp - mq)
            out["KL_rate_exact"] = 0.5 * th @ th
        print(f"Likelihood ratio ({self.kind}, {'path-level' if self.exact else 'marginal E[Z|X_t]'}) at t={t:g}")
        for k_, v in out.items():
            print(f"  {k_:>14}: {v:.5g}" if isinstance(v, (float, np.floating)) else f"  {k_:>14}: {v}")
        return out

    def lrc(self, t=1.0, start=None, grid=200, show=True, save=None):
        """Likelihood-ratio curve: q_t(x)/p_t(x) as a heatmap plus a cross-section."""
        from .viz_static import plot_lrc

        return plot_lrc(self, t=t, start=start, grid=grid, show=show, save=save)


def lr(p: Process, q: Process, seed=None) -> LikelihoodRatio:
    """Likelihood-ratio process Z = dQ/dP (simulated under P).

    >>> z = lr(bm(1, 1), bm(0, 1))      # remove the drift of a Brownian motion
    >>> z.plot_law(time_slice=10, n=25)
    >>> z.lrc()
    """
    return LikelihoodRatio(p, q, seed)


# ------------------------------------------------------------------ rare events
def rare_event(m: Process, K, T=1.0, start=None, n_max=100_000, seed=None):
    """Estimate P(X^1_T > K): naive Monte Carlo versus Esscher importance sampling."""
    if m.esscher_space is None:
        raise ValueError("rare() needs a process with an Esscher transform (bm, stocks, poisson, gamma_process).")
    s = m._start(start)
    rng = m._rng(seed)
    lo_dom, hi_dom = m.theta_domain()

    def mean_at(th):
        return m.esscher(np.array([th, 0.0])).marginal(T, s, 0).mean() - K

    a, b = max(-1.0, lo_dom + 1e-6), min(1.0, hi_dom - 1e-6)
    for _ in range(40):
        if mean_at(a) <= 0:
            break
        a = max(a * 2 if a < 0 else -1.0, lo_dom + 1e-6)
    for _ in range(40):
        if mean_at(b) >= 0:
            break
        b = min(b * 2, hi_dom - 1e-6)
    if mean_at(a) > 0 or mean_at(b) < 0:
        raise ValueError("Could not centre the tilted law at K (is K reachable?).")
    th1 = brentq(mean_at, a, b)
    th = np.array([th1, 0.0])

    X = m.sample_slices([T], n_max, rng, s)[0]
    naive = (X[:, 0] > K).astype(float)
    q = m.esscher(th)
    XQ = q.sample_slices([T], n_max, rng, s)[0]
    w = np.exp(-(m.levy_state(XQ, s) @ th) + T * m.cumulant(th))
    isest = (XQ[:, 0] > K) * w

    k = np.arange(1, n_max + 1)

    def running(v):
        mean = np.cumsum(v) / k
        var = np.maximum(np.cumsum(v * v) / k - mean ** 2, 0)
        return mean, np.sqrt(var / k)

    mn, sn = running(naive)
    mi, si = running(isest)
    exact = float(m.marginal(T, s, 0).sf(K))
    res = dict(K=K, T=T, theta=th1, n=k, naive=mn, naive_se=sn, is_=mi, is_se=si, exact=exact,
               ess=ess(w[XQ[:, 0] > K]) if np.any(XQ[:, 0] > K) else 0.0, name=repr(m))
    print(f"P(X^1_T > {K:g})  exact = {exact:.4e}   tilt theta_1 = {th1:.4f}")
    print(f"  naive MC      : {mn[-1]:.4e} +- {1.96 * sn[-1]:.2e}")
    print(f"  importance S. : {mi[-1]:.4e} +- {1.96 * si[-1]:.2e}")
    if si[-1] > 0 and sn[-1] > 0:
        print(f"  variance reduction ~ x{(sn[-1] / si[-1]) ** 2:.0f}")
    return res
