# figure_2 caption draft

Global-to-local deterministic convergence on the strongly log-concave shifted log-cosh family, built in optimizer-whitened coordinates so that $C_{\star}=I$ and the initialization $m_0=2e_1$, $C_0=I$ isolates the non-Gaussian localization from a covariance burn-in.

**(a)** Normalized objective gap on the well-conditioned control cell of the optimizer-whitened log-cosh family at $\rho=1$, each method at its frozen practical step. Values at or below the double-precision resolution of the objective are not drawn.
**(b)** Normalized objective gap on the moderately conditioned cell of the optimizer-whitened log-cosh family at $\rho=1$, each method at its frozen practical step. Values at or below the double-precision resolution of the objective are not drawn.
**(c)** Normalized objective gap on the difficult cell of the optimizer-whitened log-cosh family at $\rho=1$, each method at its frozen practical step. Values at or below the double-precision resolution of the objective are not drawn.
**(d)** Iterations to $\Delta(a_n)/\Delta(a_0)\le 10^{-6}$ over all twelve cells ($d\in\{10,50\}$, $\kappa_{\rm base}\in\{1,10,10^2\}$, $\rho\in\{0.1,1\}$), against the affine-invariant condition number $\kappa_{\star}$; marker size encodes the dimension. Every cell reaches the tolerance inside the fixed 400-iteration budget. The frozen practical step is a multiple of $1/(\beta_{\star}\max(\lambda_{0,{\star}}^{\max},1))$, and $\beta_{\star}$ ranges over $[1.04,5.09]$ here, so on the least-conditioned cells the resulting $h\gamma_{\star}$ approaches $2$, the one-step factor approaches $-1$, and the Fisher-Rao iteration count rises. That is the price of one frozen multiplier, not a property of the flow; panel (e) separates the two.
**(e)** The same cells measured in elapsed flow time $N_{10^{-6}}h$ rather than in iterations. On the eight cells with $\kappa_{\star}\ge2$ the three schemes agree to within a factor of $2.2$, which places the iteration-count differences of panel (d) in the stepsize rather than in the geometry. The flow time does not grow with $\kappa_{\star}$ over two decades of it, so the worst-case constant is far from attained on this family. The four least-conditioned cells are the exception, and for the reason given in panel (d): there the frozen multiplier overshoots, which is the cost of freezing one number rather than a property of the flow.
**(f)** Measured against predicted per-iteration contraction for both local targets, $\rho\in\{0.1,1\}$, three initial radii $r\in\{10^{-1},5\times10^{-2},10^{-2}\}$ along the slowest eigenmode of the linearized generator, and both schemes at their certified steps. The prediction is $1-h\gamma_{\star}$ for the Riemannian retraction and $1-h\gamma^{\rm KL}_{{\star},h}$, the smallest eigenvalue of the linearized KL/Bregman one-step map, for the Bregman scheme. The three radii agree to five significant figures, so the measured rate is a property of the mode and not of the initial amplitude, and the residual discrepancy is the second-order term of the discretization: it is $10^{-3}$ on the $\rho=0.1$ targets, whose certified step is five times larger, and $10^{-5}$ on the $\rho=1$ targets.

Panel data:
- (a) `results/processed/manuscript/figure_2_a.csv`
- (b) `results/processed/manuscript/figure_2_b.csv`
- (c) `results/processed/manuscript/figure_2_c.csv`
- (d) `results/processed/manuscript/figure_2_d.csv`
- (e) `results/processed/manuscript/figure_2_e.csv`
- (f) `results/processed/manuscript/figure_2_f.csv`
