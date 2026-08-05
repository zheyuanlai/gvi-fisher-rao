# Gaussian variational inference via Fisher--Rao gradient flows: numerical experiments

A float64 NumPy/SciPy reference implementation and reproducible experiment
campaign for the manuscript *Gaussian variational inference via Fisher--Rao
gradient flows*. It implements exactly the methods admitted by the scientific
protocol:

- deterministic Fisher--Rao Riemannian retraction (FR--R) and KL/Bregman (FR--KL);
- stochastic Price/Hessian--STL variants (FR--R--STL, FR--KL--STL);
- deterministic FB--GVI and minibatch S--FB--GVI of Diao, Balasubramanian, Chewi
  and Salim, implemented from Algorithm 1 of arXiv:2304.05398;
- Laplace only as a non-iterative logistic-regression approximation baseline;
- the general affine-invariant metric family of manuscript Section 5, as a
  classification-verification tool kept out of the main comparison.

It contains no BWGD, no BW--SGD, no covariance-projected Fisher--Rao scheme and no
mixed geometry; an audit gate fails the build if one of those names appears in
`src/` or `configs/`. Invalid covariance updates are recorded as failures. Only
logged roundoff-scale repairs bounded by `100 * eps * matrix_scale` are permitted,
and no run is ever silently stabilized by clipping eigenvalues or shrinking a
step.

## What the experiments test

| Experiment | Mechanism | Manuscript result |
|---|---|---|
| A | covariance burn-in in whitened coordinates | Theorem 2.9 |
| B | affine equivariance of the iterates | Proposition 2.3 |
| C | anisotropic Gaussian benchmark, `kappa = 1e9`, `kappa_star = 1` | Section 2 |
| D | strongly log-concave non-Gaussian grid | Theorems 2.14, 2.19 |
| F | exact Gaussian local region and sharp threshold | Lemma 2.24, Corollary 2.26 |
| G | near-Gaussian local spectral rate | Proposition 3.5, Theorem 3.7 |
| H | Gaussian STL pathwise cancellation | Corollary 4.20 |
| I | raw-score against STL estimator variance | Lemma 4.7 |
| J | minibatch residual floors against `B` | Theorems 4.16, 4.17 |
| K | decreasing-stepsize schedule | Theorem 4.21 |
| L | Bayesian logistic regression | applications |
| M | modal rates of the affine-invariant metric family | Section 5 |

Experiment E (fixed-step sharpness) is deliberately not run: the manuscript defers
the bump-train lower-bound construction to an appendix that does not yet exist,
and no surrogate was invented. See [blocked experiments](reports/BLOCKED_EXPERIMENTS.md).

## Stepsize fairness

Each method is swept over multiples of its **own** certified step, computed from
the closed-form optimizer-whitened curvature constants of that cell, never over a
shared numerical grid. Every comparison therefore reports the largest step at
which a method still makes progress and its best result at a fixed oracle budget,
rather than a single hand-picked step. The multiplier range extends past the
certified scale until the stability boundary is located; divergent runs are
recorded as failures and shown on the figures.

## Quick start

```bash
./scripts/bootstrap_env.sh
make test
make smoke
```

## Campaign commands

```bash
make configs                                   # regenerate the full-tier grids
CAMPAIGN_JOBS=64 OVERNIGHT_BUDGET_HOURS=8 make full
CAMPAIGN_JOBS=64 make resume                   # resume; skips matching completed jobs
make figures                                   # figures and tables from saved results only
make audit
```

Every run stores its config, config hash, numerical source hash, git state,
platform and package versions, BLAS configuration, seeds, curvature constants,
operation accounting, status and output paths under `results/manifests/`. Raw
trajectories retain individual seeds; failed seeds are never dropped.

## Numerical policy

All SPD functions use symmetric eigendecompositions; solves use Cholesky factors.
Deterministic methods within a cell share one fixed expectation design, which is
also the design on which the reference is solved, so deterministic objective gaps
are exact for the problem the algorithms actually solve. The quadrature
resolution floor of each cell and the reference's residual against an independent
four-times-larger design are recorded in
`results/tables/reference_quality.csv`.

## Documentation

- [implementation notes](reports/IMPLEMENTATION_NOTES.md)
- [numerical audit](reports/NUMERICAL_AUDIT.md)
- [campaign summary](reports/OVERNIGHT_SUMMARY.md)
- [blocked experiments](reports/BLOCKED_EXPERIMENTS.md)
- [reproduction guide](reports/REPRODUCIBILITY.md)

Manuscript figures are `results/figures/main_figure_1.{pdf,png}` through
`main_figure_5.{pdf,png}`; per-experiment figures are
`results/figures/experiment_*.{pdf,png}`. Each figure ships with the exact
processed CSV that produced it and a caption draft.
