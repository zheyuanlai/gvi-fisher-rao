# figure_a3 caption draft

Deterministic Gaussian variational inference for Bayesian logistic regression with a proper Gaussian prior $\theta\sim\mathcal N(0,\lambda^{-1}I)$, $\lambda=1$, in $d=50$ with $500$ training and $5000$ held-out points. All expectations are computed by deterministic one-dimensional quadrature over each linear predictor rather than by sampling, so a reported gap is a gap to the Gaussian variational optimum itself. BW-FB denotes the forward-backward Bures-Wasserstein scheme of Diao et al., written BW-FB for symmetry with the geometry-first Fisher-Rao labels.

**(a)** Objective gap against iteration on the hardest feature conditioning, median over 5 independently generated datasets with a min-max band. Each deterministic iteration consumes one expected gradient and one expected Hessian, so the iteration count is already a population-oracle count.
**(b)** The same trajectories against the time spent inside the update itself, excluding the diagnostics. The three methods cost 14.7 to 19.6 milliseconds per iteration, a spread of about 33 per cent, so the panel largely repeats the shape of (a). The cost is set by the expectation, which is one panelled Gauss-Legendre rule of order 48 per linear predictor and so scales as $O(nQ)$ in the $n$ observations and $Q$ nodes, rather than by the $O(d^3)$ linear algebra that distinguishes the matrix exponential of the retraction from the resolvent solve of the Bregman scheme. That distinction would become visible only at large $d$ with a cheaper expectation.
**(c)** Iterations to a relative objective gap of 1e-06, median over the 5 datasets with min-max bars; markers are offset horizontally for legibility. No method dominates: the Bures-Wasserstein baseline is fastest where the features are best conditioned and slowest where they are worst, with the crossover near $\kappa_X=10^2$. That is the expected shape, since its guarantee places the conditioning in the rate while the Fisher-Rao guarantee places it in the admissible stepsize. The Laplace approximation is a fixed point of none of the schemes and so has no iteration count; it appears in panels (a) and (b) and in the predictive table.

Panel data:
- (a) `results/processed/manuscript/figure_a3_a.csv`
- (b) `results/processed/manuscript/figure_a3_b.csv`
- (c) `results/processed/manuscript/figure_a3_c.csv`
