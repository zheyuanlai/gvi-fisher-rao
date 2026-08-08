# figure_a1 caption draft

Global-to-local deterministic convergence on the strongly log-concave shifted log-cosh family, built in optimizer-whitened coordinates so that $C_{\star}=I$ and the initialization $m_0=2e_1$, $C_0=I$ isolates the non-Gaussian localization from a covariance burn-in. These are the panels compressed out of the main-text figure.

**(a)** Normalized objective gap on the well-conditioned control cell of the optimizer-whitened log-cosh family at $\rho=1$, each method at its frozen practical step. Values at or below the double-precision resolution of the objective are not drawn.
**(b)** Normalized objective gap on the moderately conditioned cell of the optimizer-whitened log-cosh family at $\rho=1$, each method at its frozen practical step. Values at or below the double-precision resolution of the objective are not drawn.
**(c)** The same cells measured in elapsed flow time $N_{10^{-6}}h$ rather than in iterations. On the eight cells with $\kappa_{\rm base}\ge10$ the three schemes agree to within a factor of $2.2$, which places the iteration-count differences of panel (d) in the stepsize rather than in the geometry. The flow time does not grow with $\kappa_{\star}$ over the two decades the grid spans, so the worst-case constant is far from attained on this family. The four cells at $\kappa_{\rm base}=1$ are the exception, and for the reason given in panel (d): there $\beta_{\star}$ is smallest, the frozen multiplier overshoots hardest, and the Fisher-Rao flow time runs $7.6$ to $61$ times the Bures-Wasserstein one. That is the cost of freezing a single number rather than a property of the flow.

Panel data:
- (a) `results/processed/manuscript/figure_a1_a.csv`
- (b) `results/processed/manuscript/figure_a1_b.csv`
- (c) `results/processed/manuscript/figure_a1_c.csv`
