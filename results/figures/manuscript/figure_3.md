# figure_3 caption draft

Deterministic Gaussian variational inference for Bayesian logistic regression with a proper Gaussian prior $\theta\sim\mathcal N(0,\lambda^{-1}I)$, $\lambda=1$, in $d=50$ with $500$ training and $5000$ held-out points. All iterative methods share one scrambled-Sobol design for the updates, the objective evaluation and the reference solve, so the reported gaps are exact for the discretized problem the algorithms solve.

**(a)** Objective gap against iteration on the hardest feature conditioning, median over 5 independently generated datasets with a min-max band. Each deterministic iteration consumes one expected gradient and one expected Hessian, so the iteration count is already a population-oracle count.
**(b)** The same trajectories against the time spent inside the update itself, excluding the diagnostics. The three methods cost the same per iteration to within two per cent, $246$ to $250$ milliseconds, so the panel repeats the shape of~(a). On a problem of this form the cost is set by the expectation, $O(Snd)$ for $S=4096$ quadrature points and $n=500$ observations, and not by the $O(d^3)$ linear algebra that distinguishes the matrix exponential of the retraction from the resolvent solve of the Bregman scheme; the two differ by roughly three orders of magnitude in flops here. The distinction would become visible only at large $d$ with a cheap expectation.
**(c)** Terminal objective gap at the fixed iteration budget for $\kappa_X\in\{1,10^2,10^4\}$, median over the five datasets with min-max bars; markers are offset horizontally for legibility. The non-iterative Laplace approximation is a fixed point of neither scheme and is shown as a reference.

Panel data:
- (a) `results/processed/manuscript/figure_3_a.csv`
- (b) `results/processed/manuscript/figure_3_b.csv`
- (c) `results/processed/manuscript/figure_3_c.csv`
