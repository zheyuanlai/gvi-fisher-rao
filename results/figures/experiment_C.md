# experiment_C caption draft

Anisotropic Gaussian benchmark of Diao et al. ($d=10$, $\Sigma^{-1}=U\mathrm{diag}(10^{-9},\dots,1)U^\top$, $q_0=\mathcal N(0,I)$), 10 random rotations, 500 iterations. Each method is swept over multiples of its own certified step, so no geometry is forced onto another's stepsize scale. The optimizer-whitened condition number is $\kappa_{\star}=1$ here while the original-coordinate $\kappa=10^9$, which is exactly the regime the affine-invariant rate predicts the Fisher-Rao schemes to be insensitive to. Quadratic-rescued Fisher-Rao runs recover the target in one oracle query and are reported separately (median gap nan after nan query); they are excluded from the competitive curves.

Underlying data: `results/processed/experiment_C.csv`.
