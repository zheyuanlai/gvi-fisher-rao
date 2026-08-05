# Numerical audit

This report is populated alongside the validation gates.

## Gates

- SPD spectral functions and JKO eigenvalue map: implemented; automated tests in `tests/unit/`.
- Gaussian objective/Wasserstein diagnostics: implemented; automated tests in `tests/unit/`.
- Target derivatives: finite-difference validation implemented.
- Gaussian fixed points and quadratic rescue: algorithm tests implemented.
- First-order flow consistency: algorithm tests implemented.
- Stochastic unbiasedness and pathwise STL cancellation: statistical regression tests implemented.
- Affine equivariance: direct iterate-level regression test implemented.
- No silent clipping: strict failure test implemented.
- Resume/manifest logic and deterministic seed derivation: integration tests implemented.

The final observed tolerances, smoke status, and any algorithmic failures are added after execution.

