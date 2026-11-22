"""Interactive viewer (pure Python, Dash): paths on the right, density on the left.

All numbers are precomputed / cached in NumPy; the browser only displays figures.
Axes are frozen over every slice, theta and measure so nothing jumps while you slide.
"""
from __future__ import annotations

import threading
import webbrowser

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from ._utils import (ess, free_port, hist_density, hist_density_1d, quantile_range, robust_zmax)

BG, PANEL_BG, GRID, TXT, SUB = "#0b0f17", "#10151f", "#2a3140", "#d7dde8", "#9aa4b5"
PANEL_W = 330

CSS = f"""
body {{ background:{BG}; margin:0; color:{TXT}; font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif; }}
.rc-slider-rail {{ background:#2a3140 !important; }}
.rc-slider-track {{ background:#ff9f1c !important; }}
.rc-slider-handle {{ border-color:#ff9f1c !important; background:#0b0f17 !important; opacity:1 !important; }}
.rc-slider-mark-text {{ color:{SUB} !important; }}
.rc-slider-mark-text-active {{ color:{TXT} !important; }}
.rc-slider-dot {{ border-color:#2a3140 !important; background:#0b0f17 !important; }}
.rc-slider-tooltip-inner {{ background:#1b2230 !important; color:{TXT} !important; box-shadow:none !important; }}
button.rd-btn {{ background:#1b2230; color:{TXT}; border:1px solid #39435a; border-radius:8px;
  padding:8px 14px; font-size:14px; cursor:pointer; margin-bottom:8px; }}
button.rd-btn:hover {{ background:#262f42; }}
label {{ margin-right:10px; font-size:13px; color:{TXT}; }}
"""
INDEX = ("<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
         f"<style>{CSS}</style></head><body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>")


def _fine_grid(times, steps=240):
    t = np.concatenate([[0.0], np.asarray(times, float)])
    Tm = t[-1]
    parts = []
    for a, b in zip(t[:-1], t[1:]):
        k = max(4, int(round(steps * (b - a) / Tm)))
        parts.append(np.linspace(a, b, k, endpoint=False))
    return np.unique(np.concatenate(parts + [[Tm]]))


#the next few line commented because i want the time slider to start at 0 without causing an issue with the pdf
    
# def _make_times(m, time, time_slice, T):
#     Tm = m.horizon or T
#     if time is not None:
#         times = np.sort(np.atleast_1d(np.asarray(time, float)))
#     else:
#         L = int(time_slice or 10)
#         times = Tm * np.arange(1, L + 1) / L
#     if times.min() <= 0:
#         raise ValueError("slice times must be positive")
#     if m.horizon is not None and times.max() > m.horizon + 1e-12:
#         raise ValueError(f"{m!r} only lives on [0, {m.horizon:g}]")
#     return times

def _make_times(m, time, time_slice, T):
    Tm = m.horizon or T
    if time is not None:
        times = np.sort(np.atleast_1d(np.asarray(time, float)))
    else:
        L = int(time_slice or 10)
        times = Tm * np.arange(0, L + 1) / L   # starts at 0.0
    if times.min() < 0:                        # allow t = 0.0
        raise ValueError("slice times must be non-negative")
    if m.horizon is not None and times.max() > m.horizon + 1e-12:
        raise ValueError(f"{m!r} only lives on [0, {m.horizon:g}]")
    return times


# =========================================================================== scene
class Scene:
    """Everything the viewer needs: paths, cached samples, frozen axes, figure builders."""

    def __init__(self, m, start, times, n, seed, tilt, theta_range, n_density, grid,
                 colorscale, width, height):
        self.m, self.dim = m, m.dim
        self.start = m._start(start)
        self.times = np.asarray(times, float)
        self.L, self.Tmax = len(self.times), float(self.times.max())
        self.fine = _fine_grid(self.times)
        self.idx = np.searchsorted(self.fine, self.times)
        self.cs, self.tilt, self.grid = colorscale, tilt, grid
        self.height = height
        self.gw = (width - PANEL_W) // 2
        rng = np.random.default_rng(m.seed if seed is None else seed)
        self.paths = m.simulate(n, self.fine, self.start, rng)
        self.n = n
        self.big = m.sample_slices(self.times, n_density, np.random.default_rng(rng.integers(2 ** 31)),
                                   self.start, self.fine)
        self._cache = {}
        self._setup_theta(theta_range)
        self._setup_axes(rng)
        # terminal-value rank of each path -> stable colour in the un-weighted view
        term = self.paths[:, -1, 0]
        self.rank0 = np.argsort(np.argsort(term)) / max(n - 1, 1)

    # ---------------------------------------------------------------- theta / measures
    def _setup_theta(self, theta_range):
        m = self.m
        lo, hi = theta_range if theta_range else (-2.0, 2.0)
        self.theta_star = m.theta_star() if self.tilt else None
        if self.theta_star is not None:
            lo = min(lo, 1.3 * self.theta_star if self.theta_star < 0 else lo)
            hi = max(hi, 1.3 * self.theta_star if self.theta_star > 0 else hi)
        dlo, dhi = m.theta_domain()
        self.tlo, self.thi = max(min(lo, 0.0), dlo + 1e-6), min(max(hi, 0.0), dhi - 1e-6)

    def variants(self):
        m = self.m
        vs = [(0.0, "P")]
        if self.tilt:
            vs += [(self.tlo, "P"), (self.thi, "P")]
            if self.theta_star is not None:
                vs.append((float(np.clip(self.theta_star, self.tlo, self.thi)), "P"))
        elif m.measures:
            vs += [(0.0, x) for x in m.measures[1:]]
        return vs

    # ---------------------------------------------------------------- frozen axes
    def _setup_axes(self, rng):
        m, d = self.m, self.dim
        pool = [self.big.reshape(-1, d)]
        for th, ms in self.variants()[1:]:
            v = m.at(th, ms)
            pool.append(v.sample_slices(self.times, 3000, rng, self.start, self.fine).reshape(-1, d))
        inc = self.start if d == 2 else self.paths[:, 0, :].mean(axis=0)
        self.rng_ = quantile_range(np.concatenate(pool), include=inc)
        self.xr = self.rng_[0]
        self.yr = self.rng_[1] if d == 2 else None
        self.xg = np.linspace(*self.xr, self.grid)
        if d == 2:
            self.yg = np.linspace(*self.yr, self.grid)
            self.XX, self.YY = np.meshgrid(self.xg, self.yg)
        peaks = []
        for th, ms in self.variants():
            for k in range(self.L):
                peaks.append(self.density(k, th, ms, "closed").max())
        self.zmax = robust_zmax(peaks)
        self._cache.clear()
        self.center_P = np.array([m.center(tf, self.start) for tf in self.fine])

    # ---------------------------------------------------------------- densities
    def density(self, k, theta=0.0, measure="P", method="closed"):
        key = (k, round(float(theta), 6), measure, method)
        if key in self._cache:
            return self._cache[key]
        m, t = self.m, self.times[k]
        P = None

        #the next 2 line commented because i want the time slider to start at 0 without causing an issue with the pdf
    
        # if method == "closed" and self.dim == 2:
        #     P = m.at(theta, measure).pdf(t, self.XX, self.YY, self.start)

        if method == "closed" and self.dim == 2 and t > 0:
              P = m.at(theta, measure).pdf(t, self.XX, self.YY, self.start)


        if P is None:
            S = self.big[k]
            w = m.weights(t, S, self.start, theta, measure) if self.dim == 2 else None
            if self.dim == 2:
                P = hist_density(S, w, self.xg, self.yg)
            else:
                P = hist_density_1d(S[:, 0], w, self.xg)
        P = np.nan_to_num(np.asarray(P, float), nan=0.0, posinf=0.0)
        self._cache[key] = P
        return P

    # ---------------------------------------------------------------- helpers
    def _centre_path(self, upto, theta, measure):
        step = max(1, upto // 60)
        ii = np.unique(np.r_[np.arange(1, upto + 1, step), upto])
        v = self.m.at(theta, measure) if (theta != 0 or measure != "P") else self.m
        c = np.array([v.center(self.fine[i], self.start) for i in ii])
        return self.fine[ii], c

    def _colors(self, k, theta, measure):
        w = None
        if self.dim == 2 and (theta != 0 or measure != "P"):
            w = self.m.weights(self.times[k], self.paths[:, self.idx[k], :], self.start, theta, measure)
        if w is None or np.allclose(w, w[0]):
            rank = self.rank0
        else:
            rank = np.argsort(np.argsort(w)) / max(self.n - 1, 1)
        return sample_colorscale(self.cs, list(0.15 + 0.85 * rank))

    def marks(self):
        step = max(1, int(np.ceil(self.L / 12)))
        return {i: {"label": f"{t:.2f}", "style": {"color": SUB}} for i, t in enumerate(self.times)
                if i % step == 0 or i == self.L - 1}

    # ---------------------------------------------------------------- figures
    def _scene3d(self, zr, titles, aspect, eye):
        def ax(r, t):
            return dict(range=list(r), title=dict(text=t, font=dict(color=TXT, size=14)),
                        backgroundcolor="#0f1420", gridcolor=GRID, zerolinecolor=GRID,
                        showbackground=True, color=TXT, tickfont=dict(color=SUB, size=10))
        return dict(xaxis=ax(self.xr, titles[0]), yaxis=ax(self.yr, titles[1]), zaxis=ax(zr, titles[2]),
                    aspectmode="manual", aspectratio=dict(x=aspect[0], y=aspect[1], z=aspect[2]),
                    camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])))

    def _layout(self, title):
        return dict(paper_bgcolor=BG, plot_bgcolor=BG, width=self.gw, height=self.height,
                    margin=dict(l=0, r=0, t=46, b=0), showlegend=False, uirevision="keep",
                    title=dict(text=title, x=0.5, font=dict(color=TXT, size=16)), font=dict(color=TXT))

    def fig_left(self, k, theta, measure, method):
        t = self.times[k]
        P = np.minimum(self.density(k, theta, measure, method), self.zmax)
        up = self.idx[k]
        tt, c = self._centre_path(up, theta, measure)
        tag = "" if (theta == 0 and measure == "P") else f"  [theta={theta:.2f}]" if self.tilt else f"  [{measure}]"
        if self.dim == 2:
            j = np.unravel_index(np.argmax(P), P.shape)
            fig = go.Figure([
                go.Surface(x=self.xg, y=self.yg, z=P, surfacecolor=P, colorscale=self.cs, cmin=0,
                           cmax=self.zmax, showscale=False, lighting=dict(ambient=0.75, diffuse=0.6),
                           contours_z=dict(show=True, usecolormap=True, project_z=True)),
                go.Scatter3d(x=[self.xg[j[1]]], y=[self.yg[j[0]]], z=[P[j]], mode="markers",
                             marker=dict(size=6, color="white")),
                go.Scatter3d(x=c[:, 0], y=c[:, 1], z=np.zeros(len(c)), mode="lines",
                             line=dict(color="white", width=5, dash="dash")),
            ])
            fig.update_layout(scene=self._scene3d((0, self.zmax), ("X", "Y", "density"), (1, 1, 0.62),
                                                  (1.55, -1.65, 1.10)), **self._layout(f"Law of (X_t, Y_t) at t = {t:.3g}{tag}"))
        else:
            j = int(np.argmax(P))
            fig = go.Figure([
                go.Scatter(x=self.xg, y=P, mode="lines", fill="tozeroy", line=dict(color="#ff9f1c", width=3),
                           fillcolor="rgba(255,159,28,0.25)"),
                go.Scatter(x=[self.xg[j]], y=[P[j]], mode="markers", marker=dict(size=10, color="white")),
            ])
            fig.update_layout(xaxis=dict(range=list(self.xr), gridcolor=GRID, color=TXT, title="value"),
                              yaxis=dict(range=[0, self.zmax], gridcolor=GRID, color=TXT, title="density"),
                              **self._layout(f"Density of Z_t at t = {t:.3g}"))
        return fig

    def fig_right(self, k, theta, measure):
        up, t = self.idx[k], self.times[k]
        X, tt = self.paths[:, :up + 1, :], self.fine[:up + 1]
        cols = self._colors(k, theta, measure)
        _, c = self._centre_path(up, theta, measure)
        ct = self._centre_path(up, theta, measure)[0]
        traces = []
        if self.dim == 2:
            clip = lambda a, r: np.clip(a, r[0], r[1])
            for i in range(self.n):
                traces.append(go.Scatter3d(x=clip(X[i, :, 0], self.xr), y=clip(X[i, :, 1], self.yr), z=tt,
                                           mode="lines", line=dict(color=cols[i], width=3), hoverinfo="skip"))
            traces.append(go.Scatter3d(x=clip(X[:, -1, 0], self.xr), y=clip(X[:, -1, 1], self.yr),
                                       z=np.full(self.n, t), mode="markers",
                                       marker=dict(size=4, color=cols), hoverinfo="skip"))
            traces.append(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=ct, mode="lines",
                                       line=dict(color="white", width=5, dash="dash")))
            traces.append(go.Surface(x=self.xr, y=self.yr, z=np.full((2, 2), t), showscale=False,
                                     opacity=0.12, colorscale=[[0, "#8ecae6"], [1, "#8ecae6"]], hoverinfo="skip"))
            fig = go.Figure(traces)
            fig.update_layout(scene=self._scene3d((0, self.Tmax), ("X_t", "Y_t", "time"), (1, 1, 1.25),
                                                  (1.6, -1.6, 1.25)), **self._layout(f"{self.n} paths up to t = {t:.3g}"))
        else:
            for i in range(self.n):
                traces.append(go.Scatter(x=np.clip(X[i, :, 0], *self.xr), y=tt, mode="lines",
                                         line=dict(color=cols[i], width=1.6), hoverinfo="skip"))
            traces.append(go.Scatter(x=np.clip(X[:, -1, 0], *self.xr), y=np.full(self.n, t), mode="markers",
                                     marker=dict(size=7, color=cols)))
            if np.all(np.isfinite(c)):
                traces.append(go.Scatter(x=c[:, 0], y=ct, mode="lines", line=dict(color="white", dash="dash", width=3)))
            fig = go.Figure(traces)
            fig.add_hline(y=t, line=dict(color="#8ecae6", width=1, dash="dot"))
            fig.update_layout(xaxis=dict(range=list(self.xr), gridcolor=GRID, color=TXT, title="value"),
                              yaxis=dict(range=[0, self.Tmax], gridcolor=GRID, color=TXT, title="time"),
                              **self._layout(f"{self.n} paths up to t = {t:.3g}"))
        return fig

    def render(self, k, theta=0.0, measure="P", method="closed"):
        return (self.fig_left(k, theta, measure, method), self.fig_right(k, theta, measure),
                self.info(k, theta, measure, method))

    def info(self, k, theta, measure, method):
        m, t = self.m, self.times[k]
        P = self.density(k, theta, measure, method)
        v = m.at(theta, measure) if (theta != 0 or measure != "P") else m
        c = v.center(t, self.start)
        lines = [f"{m!r}", f"t = {t:.4g}   (slice {k + 1}/{self.L})",
                 "center = (" + ", ".join(f"{x:.4g}" for x in c) + ")"]
        if self.dim == 2:
            j = np.unravel_index(np.argmax(P), P.shape)
            lines.append(f"peak   = ({self.xg[j[1]]:.3g}, {self.yg[j[0]]:.3g})")
        if self.tilt:
            lines.append(f"theta = {theta:.3f}" + (f"   theta* = {self.theta_star:.3f}" if self.theta_star is not None
                                                  else "   (subordinator: drift cannot be removed)"))
        if m.measures:
            lines.append(f"measure = {measure}   density: {method}")
        if self.dim == 2 and (theta != 0 or measure != "P"):
            S = self.big[k]
            w = m.weights(t, S, self.start, theta, measure)
            lines.append(f"E[Z] = {w.mean():.3f}   ESS = {ess(w):.0f}/{len(w)}")
            if m.measures and m.esscher_space == "log":
                lines.append(f"E_w[S1]/S1(0) = {np.average(S[:, 0] / self.start[0], weights=w):.3f}")
        elif self.dim == 2 and m.measures:
            S = self.big[k]
            lines.append(f"E_P[S1]/S1(0) = {np.mean(S[:, 0] / self.start[0]):.3f}  (drifts under P)")
        if self.dim == 1:
            Z = self.big[k][:, 0]
            if not getattr(m, "as_log", False):
                lines.append(f"E[Z_t] = {Z.mean():.3f}   ESS = {ess(Z):.0f}/{len(Z)}")
                if t > 0:                                   # KL rate is 0/0 at t = 0
                    lines.append(f"-E[log Z_t]/t = {-np.mean(np.log(np.maximum(Z, 1e-300))) / t:.4g}  (KL rate)")
        return "\n".join(lines)


# =========================================================================== app
def _build_app(scene, width, height, controls, autoplay, interval, loop, title, method="closed"):
    from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

    L, m = scene.L, scene.m
    app = Dash(__name__, title=title)
    app.index_string = INDEX
    show = lambda cond: {} if cond else {"display": "none"}
    has_theta, has_meas = scene.tilt, bool(m.measures)
    gstyle = {"width": f"{scene.gw}px", "height": f"{height}px"}
    cfg = {"displayModeBar": False, "scrollZoom": False}

    tmarks = None
    if has_theta:
        pts = [scene.tlo, 0.0, scene.thi] + ([scene.theta_star] if scene.theta_star is not None else [])
        tmarks = {round(float(p), 3): {"label": f"{p:g}" if p != scene.theta_star else f"θ*={p:.2f}",
                                       "style": {"color": SUB}} for p in pts}

    side = html.Div(style={"flex": "1", "padding": "18px 14px", "boxSizing": "border-box", "overflow": "hidden"}, children=[
        html.Pre(id="info", style={"fontSize": "12.5px", "lineHeight": "1.5", "color": TXT, "whiteSpace": "pre-wrap",
                                   "marginTop": 0, "minHeight": "210px"}),
        html.Button("⏸ Pause" if autoplay else "▶ Play", id="play", className="rd-btn", style=show(L > 1)),
        html.Div(style=show(has_theta), children=[
            html.Div("Esscher θ", style={"fontSize": "13px", "color": SUB, "marginTop": "14px"}),
            dcc.Slider(id="theta", min=scene.tlo, max=scene.thi, step=(scene.thi - scene.tlo) / 400, value=0.0,
                       marks=tmarks, tooltip={"placement": "bottom", "always_visible": True}),
            html.Button("snap to drift-free θ*", id="snap", className="rd-btn",
                        style={**show(scene.theta_star is not None), "marginTop": "26px"}),
        ]),
        html.Div(style=show(has_meas), children=[
            html.Div("Measure", style={"fontSize": "13px", "color": SUB, "marginTop": "14px"}),
            dcc.RadioItems(id="measure", options=list(m.measures) or ["P"], value="P", inline=True),
        ]),
        html.Div(style=show(has_meas or has_theta), children=[
            html.Div("Density from", style={"fontSize": "13px", "color": SUB, "marginTop": "14px"}),
            dcc.RadioItems(id="method", options=[{"label": "closed form", "value": "closed"},
                                                 {"label": "reweighted paths (dQ/dP)", "value": "reweight"}],
                           value=method, inline=False),
        ]),
    ])
    slider_col = html.Div(style={"width": "96px", "padding": "30px 0", "boxSizing": "border-box", **show(L > 1)}, children=[
        dcc.Slider(id="time", min=0, max=L - 1, step=1, value=0, marks=scene.marks(), vertical=True,
                   verticalHeight=height - 60, included=True)])
    pstyle = {"width": f"{PANEL_W}px", "display": "flex", "background": PANEL_BG, "borderLeft": f"1px solid {GRID}"}
    if not controls:   # components stay alive for the callbacks but are invisible
        pstyle = {"display": "none"}
    panel = html.Div(id="panel", style=pstyle, children=[side, slider_col])
    app.layout = html.Div(style={"width": f"{width}px", "height": f"{height}px", "display": "flex",
                                 "background": BG, "margin": "0 auto", "overflow": "hidden"}, children=[
        dcc.Graph(id="left", config=cfg, style=gstyle),
        dcc.Graph(id="right", config=cfg, style=gstyle),
        panel,
        dcc.Interval(id="tick", interval=interval, disabled=not (autoplay and L > 1)),
    ])

    @app.callback(Output("left", "figure"), Output("right", "figure"), Output("info", "children"),
                  Input("time", "value"), Input("theta", "value"), Input("measure", "value"), Input("method", "value"))
    def render(k, theta, measure, method):
        k = int(k or 0)
        theta = float(theta or 0.0) if has_theta else 0.0
        measure = measure or "P"
        return (scene.fig_left(k, theta, measure, method), scene.fig_right(k, theta, measure),
                scene.info(k, theta, measure, method))

    @app.callback(Output("tick", "disabled"), Output("play", "children"), Output("time", "value"),
                  Input("play", "n_clicks"), Input("tick", "n_intervals"),
                  State("tick", "disabled"), State("time", "value"), prevent_initial_call=True)
    def play(_c, _n, disabled, k):
        k = int(k or 0)
        if ctx.triggered_id == "play":
            now_playing = disabled          # was paused -> start
            if now_playing and k >= L - 1:
                k = 0
            return (not now_playing), ("⏸ Pause" if now_playing else "▶ Play"), k
        if disabled:
            return no_update, no_update, no_update
        if k >= L - 1:
            if loop:
                return False, "⏸ Pause", 0
            return True, "▶ Play", k
        return False, "⏸ Pause", k + 1

    @app.callback(Output("theta", "value"), Input("snap", "n_clicks"), prevent_initial_call=True)
    def snap(_):
        return float(np.clip(scene.theta_star, scene.tlo, scene.thi)) if scene.theta_star is not None else no_update

    return app


def _serve(app, host, port, open_browser, block):
    url = f"http://{host}:{port}"
    print(f"radonika viewer running at {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        from IPython import get_ipython
        in_ipy = get_ipython() is not None
    except Exception:
        in_ipy = False
    if block is None:
        block = not in_ipy
    if block:
        app.run(host=host, port=port, debug=False)
    else:
        threading.Thread(target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
                         daemon=True).start()
    return app


def plot_law(m, start=None, time=None, time_slice=None, n=20, seed=None, T=1.0, tilt=False,
             theta_range=None, width=1900, height=950, controls=True, autoplay=None, interval=700,
             loop=True, n_density=20_000, grid=70, colorscale="Plasma", method="closed",
             host="127.0.0.1", port=None, open_browser=True, block=None):
    """Launch the interactive viewer.  See Process.plot_law for the parameters.

    controls=False hides the whole control panel (for screen recordings) and starts autoplay.
    """
    if time is None and time_slice is None:
        time_slice = 10
    times = _make_times(m, time, time_slice, T)
    if autoplay is None:
        autoplay = not controls
    scene = Scene(m, start, times, n, seed, tilt, theta_range, n_density, grid, colorscale, width, height)
    app = _build_app(scene, width, height, controls, autoplay, interval, loop, title=f"radonika - {m!r}", method=method)
    app._radonika_scene = scene
    return _serve(app, host, port or free_port(), open_browser, block)
