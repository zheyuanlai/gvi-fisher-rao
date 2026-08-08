# figure_2 caption draft

Global-to-local deterministic convergence on the strongly log-concave shifted log-cosh family, built in optimizer-whitened coordinates so that $C_{\star}=I$ and the initialization $m_0=2e_1$, $C_0=I$ isolates the non-Gaussian localization from a covariance burn-in.

**(a)** Normalized objective gap on the difficult cell of the optimizer-whitened log-cosh family at $\rho=1$, each method at its frozen practical step. Values at or below the double-precision resolution of the objective are not drawn.
**(b)** Iterations to $\Delta(a_n)/\Delta(a_0)\le 10^{-6}$ over all twelve cells ($N_\theta\in\{10,50\}$, $\kappa_{\rm base}\in\{1,10,10^2\}$, $\rho\in\{0.1,1\}$), against the affine-invariant condition number $\kappa_{\star}$; marker size encodes the dimension. Every cell reaches the tolerance inside the fixed 400-iteration budget. The frozen practical step is a multiple of $1/(\beta_{\star}\max(\lambda_{0,{\star}}^{\max},1))$, and $\beta_{\star}$ ranges over $[1.04,5.09]$ here, so on the least-conditioned cells the resulting $h\gamma_{\star}$ approaches $2$, the one-step factor approaches $-1$, and the Fisher-Rao iteration count rises. That is the price of one frozen multiplier, not a property of the flow; the elapsed-flow-time panel of the online appendix separates the two.
**(c)** Measured against predicted per-iteration contraction for both local targets, $\rho\in\{0.1,1\}$, three initial radii $r\in\{10^{-1},5\times10^{-2},10^{-2}\}$ along the slowest eigenmode of the linearized generator, and both schemes at their certified steps. The prediction is $1-h\gamma_{\star}$ for the Riemannian retraction and $1-h\gamma^{\rm KL}_{{\star},h}$, the smallest eigenvalue of the linearized KL/Bregman one-step map, for the Bregman scheme. The three radii agree to five significant figures, so the measured rate is a property of the mode and not of the initial amplitude, and the residual discrepancy is the second-order term of the discretization: it is $10^{-3}$ on the $\rho=0.1$ targets, whose certified step is five times larger, and $10^{-5}$ on the $\rho=1$ targets.

Panel data:
- (a) `results/processed/manuscript/figure_2_a.csv`
- (b) `results/processed/manuscript/figure_2_b.csv`
- (c) `results/processed/manuscript/figure_2_c.csv`
