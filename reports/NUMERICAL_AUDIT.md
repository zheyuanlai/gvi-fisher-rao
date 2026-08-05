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
| Minibatch floor `O(Delta t * V / B)` (Theorems 4.16, 4.17) | Experiment J, 8 cells, `B` to 64, 30 seeds | fitted `1/B` exponent near `-1.00` in every cell; floor also proportional to `Delta t` |
| Decreasing-step `O(1/N)` (Theorem 4.21) | Experiment K, 20000 iterations, 30 seeds | tail log--log slope near `-1`, no additive floor |
| Section 5 classification and modal rates | Experiment M, `(omega, tau)` grid, step refinement | predicted rates attained; residual is the `O(Delta t)` retraction bias and vanishes under refinement |

## Corrections made during the campaign

Three design errors were found and fixed by inspecting intermediate results
rather than by accepting the first output.

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
