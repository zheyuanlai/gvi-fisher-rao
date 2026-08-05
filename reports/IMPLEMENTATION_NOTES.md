# Implementation notes

## Algorithms

`src/fr_gvi/algorithms/core.py` implements only the six admitted iterative
methods. FR--R evaluates its covariance exponential through the symmetric
eigendecomposition of `I - C^(1/2) A C^(1/2)`. FR--KL solves the precision-form
system with Cholesky factors. FB--GVI follows Algorithm 1 of Diao et al.
verbatim: the forward covariance is `M C M` with `M = I - eta S`, followed by the
entropy JKO map applied eigenvalue-wise. S--FB--GVI uses the same map with its
own sampled gradient and Hessian, not the Fisher--Rao STL estimator.

FR--R--STL and FR--KL--STL use paired Gaussian samples for the Price/Hessian--STL
estimators. The raw-score variant is an estimator ablation, not a main method.
Laplace is confined to the logistic-regression cells as a non-iterative
approximation-quality baseline. No BWGD, BW--SGD, projected Fisher--Rao, or
mixed-geometry method exists anywhere in the repository; `fr_gvi.experiments.audit`
fails the build if one of their names appears in `src/` or `configs/`.

Quadratic rescue evaluates one gradient--Hessian pair, sets the exact quadratic
Gaussian solution, and records that state before any iterative step, so its
Gaussian verification reports exactly one oracle query.

`src/fr_gvi/algorithms/affine_metric.py` implements the general affine-invariant
family of manuscript Section 5, parameterized by `(omega, tau)`. It is a Section 5
verification tool and is deliberately excluded from the six-algorithm comparison.
A regression test asserts that the member `(1/2, 0)` reproduces the Fisher--Rao
retraction step to `1e-13`.

## Curvature constants

`src/fr_gvi/diagnostics/curvature.py` computes the optimizer-whitened constants
`alpha_star`, `beta_star` and `kappa_star` **in closed form** for all three target
families, with no optimization or sampling:

- Gaussian: the whitened Hessian is exactly the identity, so
  `alpha_star = beta_star = 1` regardless of the original-coordinate conditioning.
- Shifted log-cosh: with `C_star = T diag(sigma^2) T^T` the factor `R = T diag(sigma)`
  gives `R^T grad^2 V R = diag(sigma_i^2 D_ii)` with `D_ii` in `[nu_i, nu_i + rho]`,
  so both extremes are attained exactly.
- Logistic: `grad^2 V = lambda I + sum_i w_i x_i x_i^T` with `w_i` in `(0, 1/4]`, so
  the interval endpoints give the exact infimum and supremum.

Unit tests check these against a brute-force sampled maximum and verify that
`alpha_star`, `beta_star` and `kappa_star` are unchanged under an invertible
change of variables while the original-coordinate condition number is not. That
is a direct numerical check of Proposition 2.3.

## Stepsize policy

Each method is swept over multiples of its **own** certified step, never over a
shared numerical grid:

- FR--R: `1 / (2 beta_star lambda_max_star)` (Theorem 2.14).
- FR--KL: `1 / (beta_star lambda_max_star)` (Theorem 2.19).
- FR--R--STL: `1 / kappa_star`; FR--KL--STL: `1 / (8 kappa_star)` (Section 4).
- FB--GVI and S--FB--GVI: `1 / beta` (Diao et al., Corollary D.2).

`fr_gvi.experiments.grids` instantiates every cell with the same deterministic
seeds the runner will use, solves its reference, computes those constants, and
emits explicit `(step_size, normalized_step_size, certified_step_size)` triples.
The multiplier range extends far past the certified scale so that the stability
boundary is located rather than censored at the top of the grid; where the
certified step is very small (the logistic cells, where `lambda_max_star` is
large because `C_0 = I` is far from the concentrated posterior) the grid is
extended by absolute cap instead.

A run is failed when it produces NaN/Inf, an invalid covariance, an invalid
factorization, or explosive objective growth. Failures are recorded in the
manifest and plotted; they are never clipped away or silently retried at a
smaller step.

## SPD policy

Matrices are symmetrized after floating-point products. SPD square roots, inverse
square roots, logarithms, exponentials and the FB--GVI JKO map use symmetric
eigendecompositions. Cholesky solves replace dense inverses. A negative
eigenvalue is repaired only when it is no smaller than
`-100 * eps * max(spectral_scale, 1)`, and every repair is recorded in the
trajectory. Larger violations raise `AlgorithmFailure`.

`gaussian_kl_gap` computes `KL(N(m,C) || N(mu, H^{-1}))` through
`L = chol(H)` and the eigenvalues of `L^T C L`, using `x - log1p(x)` on the
shifted eigenvalues. Forming `H C` and calling `slogdet` on the product loses
positivity once both factors have condition number `1e9`, which is exactly the
regime of the Diao benchmark; the whitened form is exact to `1e-17` at the
optimizer there.

## Expectation backends

- Exact analytic expectations for Gaussian targets.
- Dimension-wise Gauss--Hermite integration (order 48 for updates, 120 for
  evaluation) for affine images of separable log-cosh targets.
- One fixed scrambled Sobol design per cell for non-separable targets, **shared**
  between the deterministic updates, the objective evaluation and the reference
  solve.
- Fresh IID Gaussian samples for stochastic updates.

Targets may expose a `weighted_hessian` fast path returning the batch-averaged
Hessian directly. For the logistic model this turns an `O(S n d^2)` contraction
into `O(S n + n d^2)`, which is what makes the `d = 100` cells tractable; a unit
test asserts it agrees with the generic per-sample average to `1e-12`.

## Reference solutions and the quadrature question

For non-separable targets the reference is the **fixed-design first-order-condition
point**, solved by damped Newton on

    E_q[grad V] = 0,        C^{-1} = E_q[grad^2 V].

This is the common fixed point of every deterministic method in the comparison:
FR--R, FR--KL and FB--GVI all consume `(E[grad V], E[grad^2 V])` and all stall
exactly there, so no geometry is privileged by the choice of reference. The point
is certified afterwards by **both** the Fisher--Rao and the Bures--Wasserstein
residual, which are at the `1e-27` level on the shared design.

On a finite quadrature design the Bonnet and Price identities hold only up to the
design error, so this point is not exactly the argmin of the reparameterized
surrogate objective. The surrogate is therefore also minimized directly by
L-BFGS on a Cholesky parameterization, the smaller of the two objective values is
used as the reference objective so every reported gap stays non-negative, and the
difference is recorded as that cell's **quadrature resolution floor**. Design
sizes were chosen from a scaling study so that the reference's residual against an
independent four-times-larger design stays near `1e-6`; both quantities appear in
`results/tables/reference_quality.csv`.

Gaussian references are exact. Log-cosh references solve the one-dimensional
Gaussian first-order conditions at high Gauss--Hermite order and transform the
result affinely, which is exact because the optimal Gaussian for a separable
target is separable.

## Manuscript mechanisms

- The decreasing schedule is implemented exactly as `h_n = 8 kappa_star / (n + n0)`
  with `n0 >= 64 kappa_star^2` and `kappa_star` taken from the closed-form
  constants of the cell.
- The near-Gaussian local operator is assembled from the manuscript's exact
  `T`, `T*` and `S` Gaussian score operators in optimizer-whitened coordinates.
  The KL/Bregman gap of Theorem 3.7(iii) is obtained from the same matrix as the
  smallest generalized eigenvalue against the weight `diag(1, ..., 1+dt, ...)`.
- Fitted local rates are compared against the **exact one-step** prediction
  `-2 log(1 - dt * gamma) / dt` rather than against `2 * gamma`, which removes the
  `O(dt)` discretization bias from the comparison; the residual bias is shown
  separately to vanish under step refinement.
- The covariance burn-in is measured in optimizer-whitened coordinates, entering
  the band when `lambda_min(C_star^{-1/2} C_n C_star^{-1/2}) >= 1/(2 beta_star)`.
- `gaussian_core_rate` evaluates the exact Gaussian-core rate `q_G` of
  Lemma 2.24 so the local Gaussian experiment can test that identity directly
  instead of testing the uniform sublevel bound as though it were a rate.

## Plotting

`src/fr_gvi/plotting/style.py` holds the shared style: Okabe--Ito colourblind-safe
colours with redundant linestyles and markers, serif text at the manuscript's
6.5-inch text width, 400 dpi PNG plus vector PDF, embedded Type 42 fonts.
`figures.py` builds one figure per experiment designed around the mechanism that
experiment tests; `main_figures.py` builds the five manuscript composites. Every
figure writes its exact processed input CSV and a caption draft. Plotting modules
are excluded from the numerical source hash, so restyling never invalidates an
expensive trajectory.

## Deliberate scope decisions

The bump-train, spiral, convex-ridge and double-well constructions are absent from
the manuscript, which defers them to an appendix that is not yet written. They
were not invented; Experiment E remains blocked and is documented in
`reports/BLOCKED_EXPERIMENTS.md`.
