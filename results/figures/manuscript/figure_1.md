# figure_1 caption draft

Gaussian structure and affine invariance. BW-FB is the forward-backward Bures-Wasserstein scheme of Diao et al., named for its geometry to match the two Fisher-Rao entries. All panels are deterministic and use the population gradient and Hessian, closed form on Gaussian targets.

**(a)** Entry time into the optimizer-whitened covariance band $\lambda_{\min}(C_n)\ge1/(2\beta_{\star})$, standard Gaussian target in $N_\theta=20$, $C_0=\lambda_0\Id$ over four decades of $\lambda_0$ at the common step $h=0.1$. The horizontal axis is the burn-in term $\log_+[1/(\beta_{\star}\lambda_{0,{\star}})]$ predicted by the global theorems, not a fit: the retraction lands on the identity to three digits (slope $0.999$) and the Bregman scheme $5$-$6\%$ above it.
**(b)** Largest discrepancy along the trajectory after the iterates of the transformed problem $x\mapsto Sx+b$ are mapped back to base coordinates. Both Fisher-Rao schemes track the base trajectory to roundoff, growing like $\varepsilon_{\rm mach}\,\mathrm{cond}(S)^2$ because the covariance is carried by the congruence $C\mapsto SCS^\top$; this is floating-point amplification through an ill-conditioned map, not a loss of equivariance. BW-FB departs by an $O(1)$ amount as soon as $S$ is non-orthogonal. The grid stops at $\mathrm{cond}(S)=10^6$ because one decade further the transported covariance is not representable in double precision.
**(c)** Normalized exact KL gap on the anisotropic Gaussian of Diao et al., $N_\theta=10$, precision eigenvalues logarithmically spaced over nine decades, each method at its own certified step. Optimizer whitening sends $\kappa=10^9$ to $\kappa_{\star}=1$: the Fisher-Rao schemes reach the double-precision floor while BW-FB stays at $0.45$, governed by the original-coordinate anisotropy. The target is built to make that separation maximal and is not evidence of dominance in general.

Panel data:
- (a) `results/processed/manuscript/figure_1_a.csv`
- (b) `results/processed/manuscript/figure_1_b.csv`
- (c) `results/processed/manuscript/figure_1_c.csv`
