# Numerical audit

This document records what was verified, how, and what remains uncertain. The
machine-readable companions are `reports/AUDIT_RESULTS.json` (manifest statuses,
forbidden-method scan, covariance positivity, pathwise covariance bands),
`reports/PLOT_AUDIT.json` (figure pairing, physical width, font embedding) and
`results/tables/reference_quality.csv` (reference certification).

## Validation gates in the test suite

The suite must pass before any campaign runs. It contains:

- SPD square root, inverse square root, log, exp and the FB--GVI JKO eigenvalue map.
- Exact Gaussian objective, Wasserstein distance, and a whitened-KL stability test
  on a precision matrix of condition number `1e9`.
- Target gradients and Hessians against finite differences.
- The fast `weighted_hessian` path against the generic per-sample average.
- Every deterministic algorithm preserving the exact Gaussian optimizer.
- One-query quadratic rescue on an arbitrary Gaussian target.
- FR--R and FR--KL matching first-order Euler expansions of the flow at small steps.
- Stochastic estimator unbiasedness at fixed `(m, C)`, and pathwise vanishing of
  the Fisher--Rao STL mean noise when the covariance is matched.
- Common-random-number pairing determinism across algorithms.
- Affine-equivariance regression at iterate level for FR--R and FR--KL.
- A no-silent-clipping test: an unstable configuration is recorded as a failure.
- Closed-form curvature constants against a brute-force sampled maximum, and their
  invariance under an invertible change of variables.
- The Section 5 metric family: `(omega, tau) = (1/2, 0)` reproduces the Fisher--Rao
  retraction to `1e-13`, and the predicted modal rates are attained to 2%.
- Config parsing, seed reproducibility, manifest atomicity and resume logic.

## What each experiment establishes

| Claim | Evidence | Outcome |
|---|---|---|
| Affine equivariance of the flow and both schemes | Experiment B, iterate-level, `K` up to `1e8` | Fisher--Rao error at roundoff, growing only in proportion to the conditioning of the map; FB--GVI `O(1)` |
| Affine invariance of `alpha_star`, `beta_star`, `kappa_star` | closed-form constants under a random change of variables | invariant to `1e-9`; original-coordinate `kappa` is not |
| Covariance burn-in `log[1/(beta_star lambda_{0,star})]` | Experiment A, 18 cells | measured/predicted ratio near one, independent of `kappa` |
| Global convergence of both discretizations | Experiments C, D, L with per-method certified sweeps | convergence to machine precision at admissible steps; certified steps conservative by orders of magnitude |
| Exact Gaussian core rate `q_G` (Lemma 2.24) | Experiment F, `d` up to 100 | measured instantaneous rate on the identity |
| Sharp Gaussian threshold (Corollary 2.26) | Experiment F, extremal initialization | measured energy equals the closed-form threshold |
| Linearized local rate (Proposition 3.5, Theorem 3.7) | Experiment G, `rho` over three decades | per-step rate settles onto the exact one-step prediction; all measurements inside the spectral bracket |
| Gaussian STL pathwise cancellation (Corollary 4.20) | Experiment H, 30 seeds | across-seed spread at floating-point scale versus `O(1)` for the native BW estimator |
| STL variance bound (Lemma 4.7) | Experiment I, direct comparison | bound holds at every state and batch size |
| Minibatch floor `O(Delta t * V / B)` (Theorems 4.16, 4.17) | Experiment J, 8 cells, `B` to 64, 30 seeds; Experiment T, four stepsizes at fixed `B` | fitted `1/B` exponent near `-1.00` in every cell, and fitted `Delta t` exponent near `+1`, so both factors of the predicted floor are measured |
| Decreasing-step `O(1/N)`, exploratory only | Experiment K, 20000 iterations, 30 seeds | tail log--log slope near `-1`, no additive floor. An earlier manuscript draft proved this; the current draft is fixed-stepsize throughout, so no figure reports it |
| Section 5 classification and modal rates | Experiment M, `(omega, tau)` grid, step refinement | predicted rates attained; residual is the `O(Delta t)` retraction bias and vanishes under refinement |
| Cancellation is a property of the estimator, not the geometry | Experiment H promoted to the manuscript tier, five stochastic methods, 30 paired seeds | three groups spanning fifteen decades: FR STL schemes at roundoff, `BBVI--STL` decaying, `Price--BBVI` and `S--FB--GVI` flat at `O(1)` |
| Published baselines reproduce their own fixed points | unit suite | `Sq--NGVI`, `Price--BBVI` and `BBVI--STL` each leave the exact Gaussian optimizing covariance fixed pathwise; `Sq--NGVI` agrees with `FR--R` to `O(h^2)` |
| The benchmark's stepsizes are computable without the optimizer | regression suite, source scan of the selection path | no optimizer-whitened constant and no reference solve on the path; every benchmark step reproduces from the recorded multiplier and base scale |
| Real posteriors stay inside the hypotheses | closed-form curvature on all five datasets | `alpha = lambda` exactly and `beta = lambda + lambda_max(X'X)/4` finite; references certify at `1e-25` residual squared |
| Dense cost is `O(d^3)` algebra over an `O(nQ)` oracle | Experiment S, `d` to 200 | the oracle dominates at small `d` and the algebra overtakes it as `d` grows; the schemes separate by their constants |

## Corrections made during the campaign

Five defects were found and fixed by inspecting intermediate results rather than
by accepting the first output.

1. **Experiment F was testing the wrong quantity.** Corollary 2.26 is a uniform
   bound on an energy sublevel, not an asymptotic rate, so comparing a fitted
   decay rate against `2 * ell_delta` failed by construction as `ell_delta` grows
   toward 1 along the trajectory. The experiment now tests the exact identity of
   Lemma 2.24 (`q_G`) and the sharp threshold, both of which hold tightly.
2. **Experiment J never reached its floor.** At the certified step, which scales
   like `1/kappa_star`, 600 iterations left the high-`kappa_star` cells inside the
   deterministic transient, so their measured floors were flat in `B` — a false
   negative for Theorems 4.16 and 4.17. With the step capped at `0.02` and 6000
   iterations, the `1/B` exponent is `-1.00` in every cell.
3. **Experiment G was measured before the asymptotic regime.** The slowest and
   fastest linearized modes differ by only a few percent, so the earlier horizon
   mixed them. The horizon was extended to the largest value float64 permits.
4. **The separable quadrature was silently wrong on the ill-conditioned cells.**
   The log-cosh integrands carry a unit-width `tanh`/`sech^2` transition inside a
   marginal whose standard deviation reaches 7 and more in the small-curvature
   coordinates. Gauss--Hermite spaces its nodes proportionally to the marginal
   width and never resolved that peak: on the widest cell it was still wrong by
   `5e-4` at order 200, and raising the order moved the answer instead of
   converging it. That shifted the fixed point of the deterministic schemes, so
   five of the 27 log-cosh cells appeared to stall at a residual of `0.03` --
   which reads as slow convergence rather than as a quadrature failure. The rule
   is now panelled Gauss--Legendre against the Gaussian density, verified at
   `1e-13` against an adaptive integrator over five orders of magnitude in the
   variance. After the fix every log-cosh reference certifies at `1.4e-13` or
   better, and all 27 cells reach machine zero.
5. **The reference's Newton solve diverged on near-separable logistic cells.**
   The undamped fixed-point iteration threw the mean out to `3e3` and then cycled,
   leaving four references with residuals between `1e5` and `1e7`. It is now
   damped by a residual-decrease condition with backtracking, and falls back to
   the directly minimized surrogate if it still stalls.
6. **A benchmark stepsize passed a screen it should have failed.** The first
   implementable-rule screen compared only the last pilot iterate with the first.
   That accepted, on `sonar`, an `FR--R` step which drove the covariance
   condition number to `4.7e10` within a few iterations before recovering; the
   resulting final trajectory needed a roundoff-scale covariance repair, which
   the manuscript audit correctly refused. A screen on the objective's excursion
   above its starting value was tried first and rejected: it also rejects the
   `FR--KL` step that is 30 times faster than any alternative on several
   datasets, whose first-step covariance contraction raises the objective
   twentyfold at a conditioning that never exceeds `62`. The screen now bounds
   the covariance conditioning at every pilot iterate, which separates the two
   cases. Two related defects were caught by an explicit transfer check before
   the campaign was launched: a multiplier calibrated on one problem instance
   diverged on the next draw of the same family, and one calibrated at the
   benchmark batch size diverged at the batch size its own panel used.

7. **The stochastic logistic comparison used an unfair step rule.** Placing all
   three methods at the same multiple of their own certified step put S--FB--GVI
   about 32 times beyond its stable range, because the Fisher--Rao certified steps
   are conservative by one to two orders of magnitude and FB--GVI's is not. It made
   no progress in any of the 18 cells. The arms were re-run at a quarter of the
   largest step each method's own deterministic counterpart was measured to
   tolerate on that cell, after which all three converge in all 18 cells.

## Known limitations

- **Experiment G, FR--KL at small `rho`.** The whitened parameter error saturates
  near `1e-26`, capping the usable horizon at about `t = 30`. With a `5%` mode gap
  that leaves a few percent of faster-mode contamination, so the measured rate sits
  slightly above the slowest-mode prediction while visibly decreasing toward it.
  This is a floating-point resolution limit, not a discrepancy with the theory, and
  the figure shows the approach rather than asserting equality.
- **Quadrature resolution floor.** For the logistic cells the deterministic methods
  solve a fixed-design discretization of the true problem. Their gaps are exact for
  that problem; the offset between it and the true objective is recorded per cell in
  `results/tables/reference_quality.csv` and is far below any gap the figures resolve.
- **Stability censoring.** Where a method survives the entire multiplier grid, the
  largest stable step is a lower bound, and those points are marked as censored on
  the figure rather than reported as if the boundary had been found.
- **Wall-clock.** Timings are CPU and environment specific, recorded for within-run
  comparison only.
- **Experiment E** is not run; see `reports/BLOCKED_EXPERIMENTS.md`.

## Failures

Failures are data, not defects. The step sweeps deliberately extend past each
method's stability boundary, and every divergent run is recorded as a failure with
its reason and iteration, retained in the manifests and plotted. No run was ever
stabilized by clipping covariance eigenvalues or by silently reducing a step. The
only permitted repair is a roundoff-scale eigenvalue correction bounded by
`100 * eps * max(spectral_scale, 1)`, and each one is written into the trajectory.
