*Under construction v0.1.0*

# radonika

> Python library to make interactive simulations to see path-laws of 2-D stochastic processes. This gives the geometry behind **Radon-Nikodym change-of-measure** for a process, in particular Girsanov's Theorem and **Esscher's Tilt** methods that helps in **Financial modelling**. 

📖 **Theory & Dev Log**: Check out GitHub repo [Development of radonika library](https://github.com/kalaapks/radonika_theory-dev) for a complete Jupyter notebook journal covering the prerequisite math (such as *Change of measure*, *Monte-Carlo Integration* as a substitute for $\mathbb{E}[.]$ etc), foundational Python simulations needed for Stochastic processes using `stochastic`, and then step-by-step development of `radonika`.

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-green)](pyproject.toml)
---

<p align="center">
  <img src="images/hero.gif" alt="hero1">
  <br>
  <b>Gamma jump process pdf evolution over time and the process itself</b>
</p>

<p align="center">
  <img src="images/stocks(T3_meas_chng).gif" alt="hero1.5">
  <br>
  <b>Stock prices under measures ℙ, ℚ, ℚ<sub>1</sub> (stock1 as numeraire), ℚ<sub>2</sub> (stock2 as numeraire) does not change the actual paths and only reweights them</b>
</p>

<p align="center">
  <img src="images/reweight_bm.gif" alt="hero2">
  <br>
  <b>Esscher Tilt via Monte Carlo Reweighting (pdf showing finite-sampling roughness).</b>
</p>
At the end of this file we have added the codes for these plots.


## The idea in one minute

There are no python libraries that helps us see the probability laws of Stochastic processes in higher dimensions. There are libraries such as `stochastics` & `aleatory` that helps with these in 1-D. This library does the same in 2-D and much more such as *Esscher's tilt* which proves to be a very strong tool when *Girsanov* can't be applied. 

Consider a process $\{(X_t, Y_t)\}_{t \in [0,1]}$ and finitely many paths of the process are realized. These paths are random and depending on where the paths end up at time $1$, we get an idea of how it would have evolved over time to end up at this configuration. If most of the paths are drawn towards a specific direction we would see that in the terminal distribution has a change in mean. However, the converse isn't true; no change in mean doesn't imply the paths haven't wandered around in the intermediate time period $(0,1)$. We may start walking, go really far away and end up back at the starting point (we see in Brownian bridge). 

![not-iid process (Brownian bridge)](images/bm_bridge_law(crop).gif)

![iid process](images/iid_normal(crop).gif)

Both Brownian bridge and i.i.d. $(0, \sigma^2)$ have the same starting and terminal distributions but one stays fixed throughout whereas the other wanders around. To distinguish between them we need to see more and more of these snapshots along the temporal direction. So, it is a good idea to "see" how the density evolves over time. That is what the above diagrams highlight.

 $1^{st}$ `hero`-diagram does the same for *Gamma Processes*. The $2^{nd}$ `hero`-diagram shows *Radon-Nikodym* change-of-measure in action - the paths themselves don't change but are reweighted to alter drift.

---

## What already existed, and what radonika adds

To our knowledge, there is no Python package that treats **two-dimensional stochastic processes** as first-class objects and lets you look at their *laws* and at *how those laws change under a change of measure*. `radonika` does that.

| Question | What radonika gives you |
|---|---|
| What does the process look like in 2-D? | Simulated paths of a pair `(X_t, Y_t)`, with optional correlation `rho` |
| How is `(X_t, Y_t)` distributed at time `t`? | The exact density where it is known in closed form, otherwise a density estimated from simulated paths |
| How does the law evolve? | An interactive 3-D viewer: paths on the right, the density surface on the left, a time slider |
| What happens if I tilt the measure? | The **Esscher transform** with a live θ slider, including the drift-free θ* |
| How do P, Q and numeraire measures differ? | A toggle between P, risk-neutral Q, and Q1 / Q2 (numeraire = stock 1 / 2) for 2-D stock prices |
| What is dQ/dP? | **Likelihood-ratio processes** (Girsanov for Brownian motion and stocks, products of density ratios for i.i.d. noise, Poisson and Gamma) and likelihood-ratio curves |
| When does dQ/dP not exist? | Clear errors when two laws are mutually singular (for example Brownian motions with different volatility), with the reason |
| What is it good for? | Rare-event estimation: naive Monte Carlo against Esscher importance sampling, with effective sample size |

The package covers Brownian motion (with drift and correlation), Brownian bridge, Poisson, Gamma and Cauchy processes, discounted 2-D stock prices under P and Q, and i.i.d. noise from any `scipy.stats` distribution.

In short: existing libraries show you **paths of one variable**; radonika shows you **the law of a pair of variables, how it moves in time, and how it changes when the probability measure changes**.

---

## Install

```bash
pip install -e .          # from this folder
```

```python
import radonika as rd        # everything lives in one namespace
```

<!-- ## Quick start

-->

---

## Tour

The library has prepared the following families of 2-dimensional processes:

- standard i.i.d. processes using normal, cauchy, exponential, uniform, student_t, laplace, lognormal 

- brownian family consisting of **2-dimensional correlated Brownian motion** with drift and brownian bridge 

<!-- ## Processes  -->
All are 2D processes naturally. However, there is an added feature that if a scalar parameter $\theta$ is passed instead of vector valued one such as $(\theta_x, \theta_y)$, it is automatically replicated to become $(\theta, \theta)$. This comes in handy, when we want to map simple processes and we do not really care about making the parameters different in two coordinates. 


| call | process | Levy | Esscher |
|---|---|---|---|
| `normal(mean, sigma, seed=None)` | iid noise per step | no | no |
| `cauchy(loc, scale, seed=None)`, `exponential(rate)`, `uniform`, `student_t`, `laplace`, `lognormal`, `iid(scipy_dist)` | iid noise | no | no |
| `bm(mu, sigma, seed=None, rho=0)` | Brownian motion with drift $\mu$ and correlation $\rho$ | yes | yes |
| `bm_bridge(sigma, end=(0,0), T=1)` | Brownian bridge | no | no |
| `poisson(rate)`, `gamma_process(a, b)` | subordinators | yes | yes |
| `cauchy_process(scale, loc=0)` | Levy, **no MGF** | yes | no (own message) |
| `stocks(mu, sigma, r, seed=None, rho=0)` | discounted GBM, prints SDE | no (log S is) | yes, on log S |
| `stocks_neutral(mu, sigma, r, seed=None)` | same under Q, prints SDE with B* | no (log S is) | yes, on log S |

Hints: your IDE shows the signature (`normal(` -> `mean, sigma, seed`), `stocks(` -> `mu, sigma, r, seed, rho, verbose`).
In Jupyter, press Shift+Tab inside the brackets.

## Methods

```python
m.plot_law(start=None, time=None, time_slice=None, n=20, seed=None, **opts)
m.tilt(...same...,  theta_range=(-2, 2))     # + horizontal Esscher theta slider

stocks(...).exchange_option()   # Margrabe, numeraire change: closed form vs MC under Q and Q1
```

`plot_law` options: `T=1`, `width=1900`, `height=950`, `controls=True`, `autoplay`, `interval=700` (ms),
`loop=True`, `n_density=20000`, `grid=70`, `method='closed'|'reweight'`, `colorscale='Plasma'`,
`port`, `open_browser=True`, `block`.

* `time=0.4` -> one slice, no slider. `time=[0.5, 0.6, 0.9, 1]` -> those slices. `time_slice=15` -> 1/15 ... 1.
* The vertical time slider sits at the far right of the window, with the theta slider / measure toggle beside it.
  `controls=False` hides the whole panel and autoplays, for clean screen recordings.
* Axes are frozen over all slices, thetas and measures, so nothing jumps while sliding.
* `n` is the number of displayed paths; densities come from closed forms or from `n_density` hidden samples.

## What the toggles do (stocks)

* **P / Q / Q1 / Q2**: the density is the exact law under the chosen measure; paths stay the same P-paths and are
  coloured by their Radon-Nikodym weight rank. Switch "Density from" to *reweighted paths* to build the density from
  the P-samples weighted by dQ/dP (Girsanov weights, numeraire weights S_T^k/S_0^k). The panel prints E[Z], ESS and
  E_w[S1]/S1(0): about 1 under Q, drifting under P.
* **theta slider** (`m.tilt`): Esscher dQ/dP = exp(theta.X_t)/M(theta)^t on a Levy process (log S for stocks).
  The "snap" button jumps to theta*, the tilt that removes the drift (theta* = (r-mu)/sigma^2 for stocks = Girsanov).


## Things to notice and ponder

* **closed-form vs reweight** 

![closed-form](images/closed_form_bm.gif)
![reweight](images/reweight_bm.gif)

The `closed-form` and `reweight` toggles appear when we code `m.tilt()`, for example, 
```python
m = rd.bm(mu=0.5, sigma=1.0, seed=1)
m.tilt(time_slice=12, n=70, theta_range=(-2, 2), seed=1)
```
For processes such as `bm`, when we change the drift along the vector $(1,1)$, it still is a `bm`. So, we have the luxury to recompute the `pdf` nicely for every single `theta` value. This is the "closed-form" and look how nice the pdfs look in the first gif. In the second one, we use Monte-Carlo sampling for the pdf and it shows up as roughness of the pdf. Before I actually created this plot, I did not even know there is so much roughness and grooves in the pdf compared to the closed-form case.

----

## CODES

Once you are done with the pip-install steps, the following lines of code will make the plots I attached in the beginning.

<p align="center">
  <img src="images/hero.gif" alt="hero1" title="" width="650" height="240">
  <br>
  <b>Gamma jump process pdf evolution over time and the process itself</b>
</p>

```python
import radonika as rd
m = rd.gamma_process(a=4, b=1.0, seed=1)
m.plot_law(start=None, time=None, time_slice=35, n=60, seed=1)
```
`time_slice` is designed to work on $[0, 1]$, so keep `time=None` or skip that argument completely. If you use `time=s`, then skip `time_slice` (here we will not get a time slider anymore, just the process on $[0, s]$). The next plot uses `time` and hence we get no temporal dynamics.

<p align="center">
  <img src="images/stocks(T3_meas_chng).gif" alt="hero1.5" title="" width="650" height="350">
  <br>
  <b>Stock prices under measures ℙ, ℚ, ℚ<sub>1</sub> (stock1 as numeraire), ℚ<sub>2</sub> (stock2 as numeraire) does not change the actual paths and only reweights them</b>
</p>

```python
import radonika as rd
m = rd.stocks(mu=0.45, sigma=0.4, r=0.03, seed=1) 
m.plot_law(start=None, time=3, n=600, seed=1)
```

<p align="center">
  <img src="images/reweight_bm.gif" alt="hero2" title="" width="650" height="320">
  <br>
  <b>Esscher Tilt via Monte Carlo Reweighting (pdf showing finite-sampling roughness).</b>
</p>

```python
import radonika as rd
m = rd.bm(mu=0.5, sigma=1.0, seed=1) 
m.tilt(time_slice=12, n=70, theta_range=(-2, 2), seed=1)
```
