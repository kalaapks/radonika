# radonika

> Python library to make interactive simulations to see path-laws of 2-D stochastic processes. This gives the geometry behind **Radon-Nikodym change-of-measure** for a process, in particular Girsanov's Theorem and **Esscher's Tilt** methods that helps in **Financial modelling**.

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-green)](pyproject.toml)
---

![](images/hero.gif)
<!-- <div align="center">

Evolution of a $2$ dimensional *Gamma Process*
</div> -->

![](images/reweight_bm.gif)

## The idea in one minute

There are no python libraries that helps us see the probability laws of Stochastic processes in higher dimentions. There are libraries such as `stochastics` & `aleatory` that helps with these in 1-D. This library does the same in 2-D and much more such as *Esscher's tilt* which proves to be a very strong tool when *Girsanov* can't be applied. 

Consider a process $\{(X_t, Y_t)\}_{t \in [0,1]}$ and finitely many paths of the process are realized. These paths are random and depending on where the paths end up at time $1$, we get an idea of how it would have evolved over time to end up at this configuration. If most of the paths are drawn towards a specific direction we would see that in the terminal distribution has a change in mean. However, the converse isn't true; no change in mean doesn't imply the paths haven't wandered around in the intermediate time period $(0,1)$. We may start walking, go really far away and end up back at the starting point (we see in Brownian bridge). 

![](images\bm_bridge_law(crop).gif)

![](images\iid_normal(crop).gif)

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

## Processes 
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

---

python lib (**under construction**) v 0.0.0
