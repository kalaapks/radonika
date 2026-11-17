# radonika

> Python library to make interactive simulations to see path-laws of 2-D stochastic processes. This gives the geometry behind **Radon-Nikodym change-of-measure** for a process, in particular Girsanov's Theorem and **Esscher's Tilt** methods that helps in **Financial modelling**. 

![hero1](images/hero.gif)
<!-- <div align="center">

Evolution of a $2$ dimensional *Gamma Process*
</div> -->

![hero2](images/reweight_bm.gif)

## The idea in one minute

There are no python libraries that helps us see the probability laws of Stochastic processes in higher dimentions. There are libraries such as `stochastics` & `aleatory` that helps with these in 1-D. This library does the same in 2-D and much more such as *Esscher's tilt* which proves to be a very strong tool when *Girsanov* can't be applied. 

Consider a process $\{X_t\}_{t \in [0,1]}$ and finitely many paths of the process are realized. These paths are random and depending on where the paths end up at time $1$, we get an idea of how it would have evolved over time to end up at this configuration. If most of the paths are drawn towards a specific direction we would see that in the terminal distribution as a change in mean. However, the converse isn't true; no change in mean doesn't imply the paths haven't wandered around in the intermediate time period $(0,1)$. We may start walking, go really far and end up back at the same point afterall (something we see in Brownian bridge). For this clarity, we need to see more and more of these snapshots along the temporal direction. So, it is a good idea to "see" how the density evolves over time. That is what the $1^{st}$ `hero`-diagram does. The $2^{nd}$ `hero`-diagram shows *Radon-Nikodym* change-of-measure in action - the paths themselves don't change but are reweighted to alter drift.


---
python lib (**under construction**) v 0.0.0
