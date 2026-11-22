"""radonika - interactive laws of 2-D stochastic processes and Radon-Nikodym visuals.

    import radonika as rd
    m = rd.stocks(0.08, 0.3, 0.02, seed=1)
    m.plot_law(time_slice=15, n=20)     # browser viewer with measure toggle (P, Q, Q1, Q2)
    m.tilt(time_slice=15)               # + Esscher theta slider
    m.analyze(t=4)                      # static matplotlib report
    z = rd.lr(rd.bm(1, 1), rd.bm(0, 1)); z.plot_law(); z.lrc()
"""
from .process import (Process, normal, cauchy, exponential, uniform, student_t, laplace, lognormal,
                      iid, bm, bm_bridge, poisson, gamma_process, cauchy_process, stocks, stocks_neutral)
from .measure import lr, LikelihoodRatio, rare_event, SingularMeasureError, NotComparableError, ess

__version__ = "0.1.0"
__all__ = [
    "normal", "cauchy", "exponential", "uniform", "student_t", "laplace", "lognormal", "iid",
    "bm", "bm_bridge", "poisson", "gamma_process", "cauchy_process", "stocks", "stocks_neutral",
    "lr", "LikelihoodRatio", "rare_event", "Process", "SingularMeasureError", "NotComparableError", "ess",
]
