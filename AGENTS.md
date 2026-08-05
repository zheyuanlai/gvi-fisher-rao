# Repository agent guidance

- Preserve the scientific exclusions in the task brief: the only Bures--Wasserstein methods are FB--GVI and S--FB--GVI.
- Never add covariance eigenvalue clipping as an algorithmic stabilization.
- Keep all numerical work in float64 and use symmetric eigendecompositions for SPD matrix functions.
- Run the numerical-validation gates before any expensive campaign.
- Treat `references/manuscript.tex` and `references/2304.05398.pdf` as the algorithmic sources of truth.
- Experiment outputs must remain resumable and retain failed seeds.

