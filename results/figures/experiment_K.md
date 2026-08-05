# experiment_K caption draft

Decreasing-stepsize schedule $\Delta t_n = 8\kappa_\star/(n+n_0)$ with $n_0=\lceil 64\kappa_\star^2\rceil$, exactly as in Theorem 4.21, run for 20000 iterations with 30 seeds on shifted log-cosh targets with $d\in\{4,8\}$, $\rho\in\{0.3,1\}$ and $B\in\{1,8\}$. Curves are sample means over seeds with standard-error bands. (a) The expected objective gap follows the predicted $O(1/N)$ decay with no additive stochastic floor. (b) The rescaled quantity $N\,\mathbb E\,\Delta(a_N)$, smoothed by a rolling median over recorded iterations. The dip and subsequent plateau is the crossover from the deterministic-contraction regime to the noise-dominated regime in which the $K/(N+n_0)$ bound of the theorem is the binding one; the annotated log-log slopes are fitted over the final decade.

Underlying data: `results/processed/experiment_K.csv`.
