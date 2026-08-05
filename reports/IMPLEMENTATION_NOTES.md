# Implementation notes

## Algorithms

`src/fr_gvi/algorithms/core.py` implements only the six admitted iterative methods. FR--R evaluates its covariance exponential through the symmetric eigendecomposition of `I - C^(1/2) A C^(1/2)`. FR--KL solves the precision-form system. FB--GVI follows Algorithm 1 of Diao et al.: the forward covariance is `M C M`, followed by the entropy JKO map applied eigenvalue-wise. S--FB--GVI uses the same map with sampled gradient and Hessian estimates.

FR--R--STL and FR--KL--STL use paired Gaussian samples for the Price/Hessian--STL estimators. The raw-score switch is an estimator ablation, not a main method. Laplace is confined to a noniterative logistic-regression reference. No BWGD, BW--SGD, projected Fisher--Rao, or mixed-geometry method is present in main configurations.

Quadratic rescue evaluates one gradient--Hessian pair, sets the exact quadratic Gaussian solution, and records that state before any iterative step. Its Gaussian verification therefore reports one query and is excluded from competitive curves.

## SPD policy

Matrices are symmetrized after floating-point products. SPD square roots, inverse square roots, logarithms, exponentials, and the FB--GVI JKO map use symmetric eigendecompositions. Cholesky solves replace dense inverses. A negative eigenvalue is repaired only when it is no smaller than `-100 * eps * max(spectral_scale, 1)`; every repair is placed in the trajectory. Larger violations raise `AlgorithmFailure` and remain visible in the manifest.

## Expectation backends

- Exact analytic expectations for Gaussian targets.
- Dimension-wise Gauss--Hermite integration for affine images of separable log-cosh targets.
- Fixed scrambled Sobol standard-normal designs for nonseparable deterministic updates.
- Independent, larger Sobol designs for evaluation and reference certification.
- Fresh IID Gaussian samples for stochastic updates.

All deterministic methods within a job share the same update design. Evaluation designs use separate seeds. Stochastic methods use paired seed derivations where common random numbers are scientifically meaningful.

## Reference optimizers

Gaussian references are exact. Log-cosh references solve one-dimensional Gaussian first-order conditions at high Gauss--Hermite order and transform the result affinely. Logistic references use a Cholesky parameterization, fixed scrambled Sobol samples, analytic reparameterization gradients, two starts, L-BFGS-B, and an independent larger evaluation design.

## Manuscript mechanisms

The decreasing schedule is implemented exactly as `h_n = 8 kappa_star / (n+n0)`, with `n0 >= 64 kappa_star^2`. The near-Gaussian local operator is assembled from the manuscript's exact `T`, `T*`, and `S` Gaussian score operators in optimizer-whitened coordinates.

Gaussian burn-in precision eigenvalues span `[1,kappa]`, matching the supplied plan. The affine test constructs `A` so `cond(A A^T)=K` and transforms both target and initialization. These two parameterization points are guarded by regression tests.

## Plotting and reporting

The named Academic Plotting Skill was not installed in the local skill catalog, so the referenced ChatGPT conversation was inspected and its publication protocol was applied directly. This led to exact 7-inch figure sizing, PDF/PNG pairing, embedded TrueType fonts, colorblind-safe colors, redundant linestyles/markers, median plus 10--90% seed bands, theoretical reference slopes, caption drafts, and a physical plot-audit command.

`src/fr_gvi/plotting/main_figures.py` builds the five manuscript-level composites. Pilot-only cells and missing constructions are visibly labeled. `main_processed.py` saves the exact core input rows and derived burn-in entries for each figure. Plotting and reporting modules are excluded from the numerical source hash, so restyling cannot invalidate expensive trajectories; all numerical algorithms, targets, factories, diagnostics, and campaign logic remain hashed.

## Deliberate deviations and scope

The bump-train, spiral, convex-ridge, and double-well constructions were not guessed because exact formulas were absent from the supplied source. Core configurations are evidence-producing pilots, not the full paper sweep. The Diao, log-cosh, and logistic panels therefore report the selected core stepsizes and budgets without claiming global superiority.

