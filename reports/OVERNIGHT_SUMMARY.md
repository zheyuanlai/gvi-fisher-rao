# Campaign summary

## Scope

The full-tier campaign covers every experiment of the plan except E, which is
deliberately not run because the manuscript defers its lower-bound construction to
an appendix that does not exist yet (see `reports/BLOCKED_EXPERIMENTS.md`).
Experiment M was added to verify the Section 5 classification, which the original
plan left as theory only.

Every comparison is stepsize-fair: each method is swept over multiples of its own
certified step, computed from the closed-form optimizer-whitened curvature
constants of that cell, and the sweep extends past the certified scale until the
stability boundary is located. Divergent runs are recorded as failures with their
reason and iteration; none is stabilized by clipping or by silently reducing a
step.

Exact run counts, statuses and failure reasons are in
`results/manifests/campaign_state.json` and the per-run manifests under
`results/manifests/full/`.

## Principal findings

These are statements about what was measured, with the caveats attached.

**Affine invariance (Proposition 2.3).** The optimizer-whitened constants
`alpha_star`, `beta_star` and `kappa_star` are unchanged to `1e-9` under a random
invertible change of variables whose original-coordinate condition number changes
by more than two orders of magnitude. At iterate level, transformed Fisher--Rao
runs match the transported reference to `2.0e-9` (FR--R) and `6.7e-9` (FR--KL) in
the worst case over `K` up to `1e8`; the residual grows in proportion to the
conditioning of the map, which is floating-point amplification rather than a loss
of equivariance. FB--GVI departs by `0.39` as soon as the map is non-orthogonal.

**Covariance burn-in (Theorem 2.9).** Over 18 cells spanning
`kappa` in `{10, 10^2, 10^3}` and `lambda_0` from `1e-2` to `1e-12`, the measured
entry time into the whitened covariance band divided by the predicted
`log[1/(beta_star lambda_{0,star})]` has median `1.024` and lies in
`[0.999, 1.064]`. The ratio is flat in `kappa`, as affine invariance requires.

**Anisotropic Gaussian benchmark.** On the target of Diao et al.
(`d = 10`, `kappa = 1e9`, but `kappa_star = 1`), the best terminal exact KL gap
over the sweep is `3.1e-17` for both Fisher--Rao schemes and `2.1e1` for FB--GVI at
500 iterations. This is the regime the affine-invariant rate predicts the
Fisher--Rao schemes to be insensitive to, and it is the most favourable case for
them; it should not be read as a general performance ranking. The quadratic rescue
recovers the target to `2.5e-17` after exactly one gradient--Hessian query, and is
reported separately rather than inside the competitive curves.

**Exact Gaussian local region (Lemma 2.24, Corollary 2.26).** Across
`d` in `{2, 10, 100}` and four initial covariance eigenvalues, the measured
instantaneous decay rate of the whitened parameter error coincides with the exact
Gaussian-core rate `q_G` along the whole trajectory, and the energy of the extremal
initialization equals the closed-form sharp threshold
`Delta_G^sharp(rho) = (rho/2 - 1 - log(rho/2))/2`.

**Near-Gaussian local rate (Proposition 3.5, Theorem 3.7).** The linearized
generator is assembled numerically from the manuscript's score operators. The
per-step contraction rate settles onto the exact one-step prediction
`-2 log(1 - dt gamma)/dt`, and every measurement lies inside the spectral bracket
between the slowest and fastest linearized modes. For FR--KL at small `rho` the
measured value sits a few percent above the slow end and is still decreasing when
the trajectory reaches the float64 resolution floor of the whitened error near
`1e-26`; this is a precision limit, not a discrepancy.

**Sticking-the-landing cancellation (Corollary 4.20).** With the covariance matched
to a Gaussian target and `B = 1`, the two Fisher--Rao STL schemes produce
across-seed objective spreads of at most `1.4e-14` over 30 seeds and
`d` in `{2, 10, 50}`, and are bit-identical across seeds at most recorded
iterations. S--FB--GVI retains a spread of `4.0`. This compares complete algorithms
together with their native estimators, not geometry alone.

**STL variance bound (Lemma 4.7).** The measured Fisher--Rao tangent variance stays
below `(2 |grad E|_a^2 + (3/2) Psi(a)) / B` at every state and batch size tested.
The STL-to-raw variance ratio collapses toward the optimizer on a Gaussian target
and plateaus on a non-Gaussian one, which is the mechanism behind the cancellation
result.

**Minibatch floors (Theorems 4.16, 4.17).** The terminal objective-gap floor decays
like `1/B` with fitted exponent within one percent of `-1` in every cell of the
`d x kappa x rho` grid. Its dependence on the step is steeper than linear and lies
between the `Delta t` and `Delta t^2` references, which is what the STL structure
predicts: the Gaussian-core part of the noise is itself proportional to
`C_n - C_star`, whose stationary size is already `O(Delta t)`. The quadratic rescue
leaves the floor unchanged and instead removes the transient.

**Decreasing stepsizes (Theorem 4.21).** With the exact schedule
`Delta t_n = 8 kappa_star / (n + n_0)`, `n_0 = ceil(64 kappa_star^2)`, run for 20000
iterations with 30 seeds, the tail log--log slope of the expected gap is `-0.98`
over the final decade and `-1.01` over the final half, with no additive floor.

**Section 5 classification.** Over 48 members `(omega, tau)` of the classified
family and `N` in `{2, 5, 10}`, the traceless and trace covariance modes decay at
the predicted `1/(2 omega)` and `1/(2 (omega + tau N))` with maximum relative error
`5.1e-3` and `1.0e-2` at the finest step. The residual is the `O(Delta t)` bias of
the retraction and vanishes under step refinement.

**Log-cosh application.** Over the 27 cells of the
`d x kappa_base x rho` grid, with each method at its own best swept step and the
same 400 gradient--Hessian pairs, the better of the two Fisher--Rao schemes reaches
machine zero in all 27 cells and FB--GVI in 3; FB--GVI's best gap is larger in 25 of
27 cells. The median largest stable step, as a multiple of each method's own
certified step, is 32 for FR--R, 16 for FR--KL and 2 for FB--GVI, so the certified
Fisher--Rao steps are conservative by one to two orders of magnitude. Per-cell
numbers are in `results/tables/stepsize_summary.csv`.

Two qualifications belong with that comparison. FB--GVI's certificate
`eta <= 1/beta` assumes `beta^{-1} I <= Sigma_0`, which the shared initialization
`C_0 = 0.5 I` fails in 17 of the 46 swept cells, and every FB--GVI divergence at or
below its certified step falls in those cells. And the objective gap is measured in
the Fisher--Rao problem's own terms; it is the quantity both geometries minimize,
but a fixed oracle budget is not the only reasonable accounting.

**Logistic application.** Both Fisher--Rao schemes reach machine-zero gaps at
admissible steps; FB--GVI is several orders of magnitude behind at the same budget
on the ill-conditioned cells. Held-out predictive negative log-likelihood converges
to the same value for every iterative method, as it must once they share an
optimizer; the Laplace approximation sits at a different point, with a nonzero
Gaussian-VI objective gap.

The stochastic logistic arms were run twice. The first pass put all three methods
at the same multiple of their own certified step. Because the Fisher--Rao certified
steps are conservative by one to two orders of magnitude while FB--GVI's is not,
that placed S--FB--GVI roughly 32 times beyond its stable range and it made no
progress in any of the 18 cells — an artefact of the step rule, not a property of
the method. The comparison was re-run in the `Lstoch_*` cells with each method at a
quarter of the largest step its own deterministic counterpart was measured to
tolerate on that exact cell. All three then converge in all 18 cells; the reported
figures use the corrected runs and the first pass is excluded.

## Corrections made during the campaign

Three design errors were found by inspecting intermediate results and fixed; they
are described in `reports/NUMERICAL_AUDIT.md`. In brief: Experiment F was
comparing a fitted rate against a uniform sublevel bound; Experiment J's horizon
was too short for the high-`kappa_star` cells to leave their deterministic
transient, which produced a false negative on the `1/B` floor law; and Experiment
G was measured before its asymptotic regime.

## Artifacts

- Manuscript figures: `results/figures/main_figure_1.{pdf,png}` through
  `main_figure_5.{pdf,png}`.
- Per-experiment figures: `results/figures/experiment_*.{pdf,png}`.
- Caption drafts alongside each figure as `.md`, and the exact processed input CSV
  under `results/processed/`.
- Tables: `results/tables/stepsize_summary.*`, `headline_summary.*`,
  `reference_quality.*`.
- Audits: `reports/AUDIT_RESULTS.json`, `reports/PLOT_AUDIT.json`.

## Reproducing or extending

```bash
make test
CAMPAIGN_JOBS=64 make full      # resumable; skips jobs whose hashes match
make figures                    # regenerates every figure and table from saved data
make audit
```

Editing a hashed source file invalidates previous runs by design. Editing a
plotting module does not, so figures can be restyled without recomputation.
