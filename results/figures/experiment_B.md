# experiment_B caption draft

Affine equivariance (Proposition 2.3). A Gaussian target and its initialization are transformed by an invertible affine map of conditioning $K$, and the transformed iterates are compared with the iterates of the untransformed run. Both Fisher--Rao schemes agree with the transported reference to within roundoff across eight decades of $K$. Their residual grows in proportion to the conditioning of the change of variables, from $10^{-15}$ at $K=1$ to $10^{-9}$ at $K=10^8$, which is floating-point amplification through an ill-conditioned map rather than a loss of equivariance. FB--GVI, whose Bures--Wasserstein geometry is not affine invariant, departs by an $O(1)$ amount as soon as the map is non-orthogonal. This is an iterate-level identity test, not a cross-geometry performance comparison.

Underlying data: `results/processed/experiment_B.csv`.
