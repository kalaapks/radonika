"""Static matplotlib figures: analyze, lrc, rare."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ._utils import hdr_levels, hist_density, quantile_range

BG, PANEL, TXT, GRID = "#0b0f17", "#141a26", "#d7dde8", "#2a3140"
SLICE_COLORS = ["#00e5ff", "#b388ff", "#ffd54f", "#69f0ae", "#ff9100", "#ff80ab"]


def _style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#9aa4b5", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, lw=0.5, ls=":")


def _fine(times, steps=240):
    return np.unique(np.concatenate([np.linspace(0, max(times), steps + 1), times]))


def _density(m, t, xg, yg, S, start):
    X, Y = np.meshgrid(xg, yg)
    P = m.pdf(t, X, Y, start)
    if P is None:
        P = hist_density(S, None, xg, yg, smooth=2.0)
    return np.nan_to_num(P)


def analyze(m, t=4, start=None, times=None, T=1.0, n=400, n_paths=60, seed=None, save=None,
            show=True, grid=220):
    """Top row: scatter + faint paths at t slices; bottom row: HDR-contour evolution."""
    if m.dim != 2:
        raise ValueError("analyze() works on 2-D processes")
    k = int(t)
    if not 1 <= k <= 6:
        raise ValueError("t (number of slices) must be between 1 and 6")
    Tm = m.horizon or T
    times = np.array(times, float) if times is not None else Tm * np.arange(1, k + 1) / k
    times = np.sort(times)
    k = len(times)
    s = m._start(start)
    rng = m._rng(seed)
    fine = _fine(times)
    paths = m.simulate(n_paths, fine, s, rng)
    big = m.sample_slices(times, max(n, 20000), rng, s, fine)
    rngs = quantile_range(np.concatenate(list(big)), include=s)
    xg, yg = np.linspace(*rngs[0], grid), np.linspace(*rngs[1], grid)
    cell = (xg[1] - xg[0]) * (yg[1] - yg[0])
    centers = np.array([m.center(tf, s) for tf in fine])

    with plt.rc_context({"font.family": "DejaVu Sans"}):
        fig = plt.figure(figsize=(17, 10), facecolor=BG)
        gs = fig.add_gridspec(2, k, height_ratios=[1, 1.25], hspace=0.28, wspace=0.12,
                              left=0.05, right=0.985, top=0.92, bottom=0.07)
        for j, tj in enumerate(times):
            ax = fig.add_subplot(gs[0, j])
            _style(ax)
            c = SLICE_COLORS[j]
            idx = np.searchsorted(fine, tj)
            for p_ in paths:
                ax.plot(p_[:idx + 1, 0], p_[:idx + 1, 1], color="#8892a6", lw=0.5, alpha=0.35)
            ax.scatter(big[j, :n, 0], big[j, :n, 1], s=7, color=c, alpha=0.55, lw=0)
            cc = m.center(tj, s)
            ax.scatter(*cc, s=110, color=c, edgecolor="white", lw=1.8, zorder=5)
            ax.set_xlim(rngs[0]); ax.set_ylim(rngs[1])
            ax.set_title(f"Slice t = {tj:g}", color=c, fontsize=12, fontweight="bold")
            ax.set_xlabel(r"$X_{1,t}$", color=TXT)
            if j == 0:
                ax.set_ylabel(r"$X_{2,t}$", color=TXT)
            else:
                ax.set_yticklabels([])
        ax = fig.add_subplot(gs[1, :])
        _style(ax)
        handles = []
        for j, tj in enumerate(times):
            P = _density(m, tj, xg, yg, big[j], s)
            lv90, lv50 = hdr_levels(P, cell, (0.9, 0.5))
            levels = sorted({lv90, lv50})
            ax.contour(xg, yg, P, levels=levels, colors=SLICE_COLORS[j],
                       linewidths=[1.1, 2.6][:len(levels)] if len(levels) > 1 else [2.0])
            handles.append(Line2D([], [], color=SLICE_COLORS[j], lw=2.5, label=f"Slice t = {tj:g}"))
        ax.plot(centers[:, 0], centers[:, 1], color="white", ls=":", lw=2.2)
        handles.append(Line2D([], [], color="white", ls=":", lw=2.2, label=r"Drift path (mean / median)"))
        ax.set_xlim(rngs[0]); ax.set_ylim(rngs[1])
        ax.set_xlabel(r"$X_1$", color=TXT); ax.set_ylabel(r"$X_2$", color=TXT)
        ax.set_title(f"CONTOUR EVOLUTION  (thin: 90% region, thick: 50% region)", color=TXT,
                     fontsize=13, fontweight="bold")
        ax.legend(handles=handles, facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, loc="upper left")
        fig.suptitle(f"{m!r}  -  law of (X_t, Y_t)", color=TXT, fontsize=15, fontweight="bold")
    if save:
        fig.savefig(save, dpi=160, facecolor=BG)
    if show:
        plt.show()
    return fig


def plot_lrc(z, t=1.0, start=None, grid=200, show=True, save=None):
    """Likelihood-ratio curve q_t/p_t: heatmap of log10 ratio + cross-section through the centres."""
    p, q = z.p, z.q
    s = p._start(start)
    S = np.concatenate([p.sample(t, 6000, s, 1), q.sample(t, 6000, s, 2)])
    rg = quantile_range(S, include=s)
    xg, yg = np.linspace(*rg[0], grid), np.linspace(*rg[1], grid)
    X, Y = np.meshgrid(xg, yg)
    P, Q = p.pdf(t, X, Y, s), q.pdf(t, X, Y, s)
    if P is None or Q is None:
        raise ValueError("lrc needs closed-form densities for both processes")
    with np.errstate(all="ignore"):
        R = np.log10(Q / P)
    lim = np.nanpercentile(np.abs(R[np.isfinite(R)]), 95) if np.isfinite(R).any() else 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor=BG, gridspec_kw=dict(width_ratios=[1.25, 1]))
    for a in (a1, a2):
        _style(a)
    im = a1.pcolormesh(xg, yg, np.clip(R, -lim, lim), cmap="coolwarm", shading="auto", vmin=-lim, vmax=lim)
    a1.contour(xg, yg, R, levels=[0.0], colors="white", linewidths=1.6)
    a1.contour(xg, yg, P, levels=hdr_levels(np.nan_to_num(P), (xg[1] - xg[0]) * (yg[1] - yg[0]), (0.5,)),
               colors="#00e5ff", linewidths=1.5, linestyles="--")
    a1.set_title(rf"$\log_{{10}}\, q_t/p_t$ at $t={t:g}$   (white: ratio = 1, dashed: 50% of P)", color=TXT)
    a1.set_xlabel(r"$X_1$", color=TXT); a1.set_ylabel(r"$X_2$", color=TXT)
    cb = fig.colorbar(im, ax=a1); cb.ax.tick_params(colors="#9aa4b5")
    cy = p.center(t, s)[1]
    jy = np.argmin(np.abs(yg - cy))
    a2.plot(xg, P[jy], color="#00e5ff", lw=2, label="p (P)")
    a2.plot(xg, Q[jy], color="#ffd54f", lw=2, label="q (Q)")
    a2.set_xlabel(rf"$X_1$  (cross-section at $X_2={yg[jy]:.2f}$)", color=TXT)
    a2.set_ylabel("density", color=TXT)
    b = a2.twinx()
    with np.errstate(all="ignore"):
        b.plot(xg, Q[jy] / P[jy], color="#ff80ab", lw=2, ls="--", label="dQ/dP = q/p")
    b.set_yscale("log"); b.tick_params(colors="#ff80ab")
    b.set_ylabel("likelihood ratio q/p", color="#ff80ab")
    h1, l1 = a2.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, facecolor=PANEL, edgecolor=GRID, labelcolor=TXT)
    a2.set_title("Radon-Nikodym derivative along a cross-section", color=TXT)
    fig.suptitle(f"Likelihood-ratio curve  |  P = {p!r}   Q = {q!r}", color=TXT, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, facecolor=BG)
    if show:
        plt.show()
    return fig


def plot_rare(res, show=True, save=None):
    """Running naive vs importance-sampling estimate of a tail probability with 95% bands."""
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=BG)
    _style(ax)
    n = res["n"]
    for key, sek, c, lab in (("naive", "naive_se", "#ff80ab", "naive Monte Carlo"),
                             ("is_", "is_se", "#00e5ff", "Esscher importance sampling")):
        m, se = res[key], res[sek]
        ax.plot(n, m, color=c, lw=2, label=lab)
        ax.fill_between(n, np.maximum(m - 1.96 * se, 1e-300), m + 1.96 * se, color=c, alpha=0.2)
    ax.axhline(res["exact"], color="white", ls=":", lw=1.8, label=f"exact = {res['exact']:.3e}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(res["exact"] / 30, res["exact"] * 30)
    ax.set_xlabel("number of simulated paths", color=TXT)
    ax.set_ylabel(rf"estimate of $P(X^1_T > {res['K']:g})$", color=TXT)
    ax.set_title(f"Rare event: naive vs tilted (theta_1 = {res['theta']:.3f})   {res['name']}", color=TXT, fontweight="bold")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, facecolor=BG)
    if show:
        plt.show()
    return fig
