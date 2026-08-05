# Numerical audit

## Validation result

The final test run reports `38 passed`. The active smoke-plus-core campaign reports 164 completed, zero failed, zero interrupted, zero pending, and two source-of-truth skips. `reports/AUDIT_RESULTS.json` contains no audit error, and `reports/PLOT_AUDIT.json` contains no error or warning across 16 PDF/PNG pairs.

## Algorithm and diagnostic gates

- FR--R and FR--KL: exact Gaussian fixed points, first-order flow consistency, strict SPD behavior, and one-query quadratic rescue are covered by unit and regression tests.
- FR--R--STL and FR--KL--STL: sampled Price/Hessian updates, common-random-number behavior, estimator unbiasedness, and exact Gaussian pathwise cancellation are tested. The core cancellation spread is at floating-point scale.
- FB--GVI and S--FB--GVI: the forward covariance map and entropy JKO eigenvalue map follow Diao et al.; Gaussian fixed points and stochastic updates are tested.
- Laplace: available only as a noniterative logistic approximation baseline and absent from competitive algorithm enums outside that use.
- Gaussian KL, Gaussian W2, optimizer-relative mean/covariance errors, covariance eigenvalue bands, and separate Fisher--Rao/Bures--Wasserstein residuals are evaluated independently.
- The affine experiment transforms target and initialization together. Automated iterate-level tolerances reach `3.83e-13` covariance and `4.54e-13` mean for the FR methods at `cond(A A^T)=1e4`.
- Target gradients and Hessians are finite-difference checked. Exact Gaussian, Gauss--Hermite, fixed scrambled Sobol, and fresh-IID backends have regression coverage.
- Invalid SPD updates raise `AlgorithmFailure`; only roundoff repairs bounded by `100 eps max(scale,1)` are allowed and recorded. No active core job failed this gate.
- Campaign integration tests cover `SeedSequence` derivation, paired streams, atomic manifests, config/source hashes, budget behavior, and resumability.

## Cross-checks against theory-specific diagnostics

- Recorded hard-start covariance entry times track `log[1/(beta lambda_0)]` at the two completed core scales.
- Gaussian rescue rows now contain one oracle pair and machine-precision objective gaps; no redundant post-rescue update is counted.
- The local score operator uses the manuscript's `T`, `T*`, and `S` definition in optimizer-whitened coordinates. The one completed non-Gaussian core cell predicts `2 gamma_star=1.996`, bracketed by fitted FR rates 1.885 and 2.166.
- Decreasing-step schedules are exactly `h_n=8 kappa_star/(n+n0)` with `n0>=64 kappa_star^2`; fitted tail exponents are -0.903 and -0.891.

## Plot audit

The plotting protocol from the referenced Academic Plotting conversation influenced the final output: Matplotlib-first styling, colorblind-safe colors, redundant line/marker encodings, seed medians with 10--90% bands, exact manuscript width, paired PDF/PNG files, embedded fonts, and caption drafts.

All 16 figures are exactly 7.0 inches wide. PDFs are 504 points wide; PNGs are 1540 pixels wide at 220 dpi. Font embedding checks pass. The five composite manuscript figures were visually inspected at rasterized resolution; two clipped titles were shortened and rerendered. The audit tool still marks visual inspection as a separate human gate for regenerated artifacts.

## Unresolved concerns

- Experiment E is blocked because the bump-train formula and constants are missing. The skip is intentional and manifested at both tiers.
- Full stepsize, condition-number, dimensional, feature-conditioning, and 30-seed grids remain unrun. Single-cell stability/performance panels are labeled as pilots.
- The logistic reference uses finite QMC and has squared FR residual `2.34e-5` and squared BW residual `2.99e-4`. Logistic gaps are approximate relative gaps.
- Wall-clock numbers are CPU and environment specific. They are stored for within-run comparison, not universal performance claims.

