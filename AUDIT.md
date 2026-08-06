# Independent manuscript-readiness audit

Date: 2026-08-06

## Verdict

**Not manuscript-ready. Do not start another full campaign or cite the current figures in the manuscript.**

The reduction to three deterministic experiment groups is directionally correct, and most of the requested exclusions are respected. However, the current artifacts do not constitute one frozen, reproducible experiment campaign. There are also scientific problems in the affine-equivariance and logistic-regression results, a permissive reference-certification gate, and a failing smoke gate.

Claude Code should treat the P0 findings below as release blockers. Fix the validation and protocol issues first, freeze a clean revision, and only then rerun the reduced campaign.

## Scope of this audit

I checked:

- the manuscript configuration tree, run manifests, raw results, processed data, figures, tables, and provenance files;
- the implementations of FR--R, FR--KL, FB--GVI, reference construction, pilot selection, manuscript plotting, and manuscript auditing;
- consistency with `references/manuscript.tex`, `references/2304.05398.pdf`, the requested theory-first experiment plan, and `AGENTS.md`;
- the numerical-validation gates and the new test coverage;
- all three generated manuscript figures at approximately their intended display size;
- factual claims in `reports/NUMERICAL_SECTION.tex`, `reports/MANUSCRIPT_PROTOCOL.md`, and the reproducibility documentation.

I did not rerun the expensive campaign. The existing raw results were sufficient to audit it.

## Gate results

| Gate | Result | Consequence |
| --- | --- | --- |
| `make test` | Pass: 82 tests | Unit/regression tests are green, but important release invariants are not covered. |
| `make smoke` | **Fail** | Release blocker. `scripts/run_smoke.sh` passes `--raw-root`, which `src/fr_gvi/plotting/figures.py` does not accept. |
| `make manuscript-audit` | Exit 0 with warnings | Not a sufficient pass: it explicitly accepts a failed trajectory and covariance repairs. |
| Plot width/font audit | Pass | The three PDFs have acceptable physical width and embedded fonts. |
| Manual figure audit | Mixed | Figures are readable, but Figure 1(b) contains invalid repaired/failed data and Figure 3 is not supported by an independent expectation/reference design. |

Running `make smoke` regenerated tracked smoke outputs. Those audit-induced changes were restored; no pre-existing experiment work was reverted.

## P0: release blockers

### 1. The 134 trajectories were not produced from one frozen code state

The 134 manifests contain four distinct `code_hash` values:

- all 60 logistic trajectories use one hash;
- the 36 global log-cosh, 12 local log-cosh, and 3 anisotropic-Gaussian trajectories use another;
- the 15 affine-equivariance trajectories use a third;
- the 8 Gaussian burn-in trajectories use a fourth.

All 134 manifests report `git_dirty: true`. They share a recorded Git commit, but the differing source hashes show that the code changed between experiment groups. There is no `numerics-protocol-*` tag. This is not one reproducible frozen campaign.

The configuration hashes do match the corresponding manifests, and all referenced raw files exist. The defect is source provenance, not missing data.

Relevant code and artifacts:

- `src/fr_gvi/experiments/campaign.py:847-878`
- `src/fr_gvi/plotting/manuscript_figures.py:75-83`
- `results/manifests/manuscript/`
- `results/figures/manuscript/figure_1.json`
- `results/figures/manuscript/figure_2.json`
- `results/figures/manuscript/figure_3.json`

Required fix:

1. Complete every code, protocol, test, and documentation correction.
2. Run `make test` and `make smoke` successfully.
3. Commit the exact source and selected protocol, require a clean tree, and create the protocol tag.
4. Regenerate every final trajectory from that one revision.
5. Make the manuscript audit fail unless every manifest has the expected unique commit/code hash and `git_dirty` is false.
6. Put the source hash, not only the commit and dirty flag, in each figure provenance JSON.

### 2. Figure 1(b) includes one failed trajectory and two covariance-repaired trajectories

At `cond(S)=10^8`:

- FR--R fails with `LinAlgError: Matrix is not positive definite` after only a few updates;
- FR--R records one covariance repair before failing;
- FB--GVI records two covariance repairs;
- the plotted maximum affine errors are approximately 46.4 for FR--R and 48.0 for FB--GVI.

The repairs are not negligible state-preserving roundoff adjustments. For example, a negative eigenvalue is replaced by a scale-dependent positive value of order 223--234. The plotted FR--R maximum also comes from a short failed trajectory, while the other methods are evaluated over complete trajectories. The panel therefore does not make a valid method comparison.

The manuscript audit currently downgrades this to a warning, and the configuration generator preregisters the failure as expected:

- `src/fr_gvi/experiments/manuscript.py:357-367`
- `src/fr_gvi/experiments/manuscript_audit.py:77-124`
- `src/fr_gvi/linear_algebra/spd.py:41-63`
- `results/processed/manuscript/figure_1_b.csv`
- `reports/MANUSCRIPT_AUDIT.json`

This conflicts with the final-figure requirement of no clipping, repairs, or backtracking and with the repository rule never to use covariance eigenvalue clipping as algorithmic stabilization.

Required fix:

- The final audit must fail on any failed, truncated, repaired, clipped, or backtracked final trajectory.
- Do not repair the state to make the `10^8` point finish.
- Either implement a stable affine-equivariance calculation that completes without modifying the algorithmic state, or lower the largest transformation condition number to the range that float64 can represent reliably (for example, stop at `10^6`) and explain the numerical limit.
- Regenerate Figure 1(b). Do not plot partial failed trajectories as comparable observations.

### 3. The logistic experiment does not use independent update, evaluation, and reference designs

The requested protocol calls for a common deterministic update design, an independent finer evaluation design, and a still finer reference design. Instead:

- the configs set update, evaluation, and reference counts to 4096;
- `build_expectation_engines` returns the update engine as the evaluation engine;
- the reference is solved/evaluated on that shared finite design;
- an independent 4x design is used only for a transfer diagnostic, not for the reported trajectory objective and gradient.

Relevant locations:

- `src/fr_gvi/experiments/manuscript.py:524-535`
- `src/fr_gvi/experiments/manuscript.py:590-593`
- `src/fr_gvi/experiments/campaign.py:189-211`
- `src/fr_gvi/experiments/campaign.py:920-943`
- `src/fr_gvi/experiments/reference.py:314-348`
- `reports/MANUSCRIPT_PROTOCOL.md:211-225`

This matters numerically. The recorded independent-design transfer residual is about `8.6e-4` to `3.4e-3`, and the transfer objective discrepancy reaches about `9.2e-3`. These discrepancies are many orders of magnitude larger than the `1e-6` convergence tolerance and the small gaps shown in Figure 3. The existing results establish convergence for a finite-design surrogate, not the requested independently evaluated Gaussian-VI objective.

Required fix:

- Build genuinely separate update, evaluation, and reference expectation engines with independent seeds/designs.
- Use the same update and evaluation designs across methods for a dataset, while keeping update and evaluation designs independent from each other.
- Use the finer independent reference/evaluation design for the reported objective gap and Fisher--Rao residual.
- Reuse one certified reference per dataset across methods rather than recomputing it four times.
- Regenerate all logistic results, tables, and Figure 3.

### 4. Reference certification does not implement the stated acceptance criterion

The requested rule is direct: the reference Fisher--Rao residual must be at least two orders of magnitude below the smallest reported objective gap. The audit instead defines

```text
objective_error = 0.5 * residual^2
```

and compares that inferred number to the gap. No curvature constant or proved residual-to-objective conversion is supplied. This is not the requested criterion and does not provide a general rigorous error bound.

The audit also records, but does not gate on, the much larger independent-design transfer residual and transfer objective discrepancy. Negative objective gaps are then used to define a plotting "reference floor", which hides reference mismatch instead of rejecting it.

Relevant locations:

- `src/fr_gvi/experiments/manuscript_audit.py:137-190`
- `src/fr_gvi/experiments/reference.py:334-336`
- `src/fr_gvi/experiments/campaign.py:936-943`
- `src/fr_gvi/plotting/manuscript_figures.py:777-792`
- `results/tables/manuscript_reference_certification.csv`
- `reports/NUMERICAL_SECTION.tex:46-51`

There is an additional consistency bug: `solve_reference` may select the lower-objective candidate, but `run_config` subsequently overwrites `reference.objective` with the objective of the fixed-point candidate. This contributes to negative reported gaps.

Required fix:

- Preserve and evaluate the actually selected reference state consistently.
- Gate directly on the stated residual margin, or implement a proved residual-to-objective bound including all required constants and state it precisely.
- Gate on independent-design transfer residual/objective discrepancy.
- Reject material negative objective gaps; do not derive a resolution floor from the largest negative excursion across methods.
- Make reference certification a hard failure before plotting.

### 5. The stepsize protocol was changed after inspecting final target behavior

The requested protocol specifies one pilot target, `h = eta * h_cert`, and the largest admissible multiplier frozen across final non-Gaussian experiments. The implementation instead:

- uses two pilot families;
- defines a different scale involving `beta_star` and the initial covariance;
- selects the fastest admissible multiplier rather than simply the largest admissible multiplier;
- documents that the prescribed scheme diverged on six final grid cells and that alternative scales were tried after observing this behavior.

Relevant locations:

- `src/fr_gvi/experiments/manuscript.py:51-88`
- `src/fr_gvi/experiments/manuscript.py:168-232`
- `src/fr_gvi/experiments/pilot.py:174-210`
- `reports/MANUSCRIPT_PROTOCOL.md:161-225`
- `configs/manuscript/selected_steps.json`

The modified scheme may be defensible as a new practical protocol, but it is exploratory rather than preregistered. It cannot be described as the requested frozen pilot protocol.

Required fix:

- Preferred: restore the specified single pilot and certified-step multiplier rule.
- If that rule is genuinely unsuitable, write a new protocol with mathematical justification, freeze it before looking at regenerated final results, and label the prior results exploratory. Do not retrofit the protocol to the already observed final grid.
- Keep theorem-verification panels on theorem-compatible steps exactly as requested.
- Correct any claim that the FB multiplier is universally the largest theorem-admissible step unless the implemented scale is exactly `1/beta` in every relevant cell.

### 6. The smoke gate fails

`make smoke` reaches the plotting stage and fails because:

```text
figures.py: error: unrecognized arguments: --raw-root ...
```

Relevant locations:

- `scripts/run_smoke.sh:11`
- `src/fr_gvi/plotting/figures.py:1377-1379`

Required fix: reconcile the CLI and the script, add coverage for this invocation, and require `make smoke` to pass before any final campaign.

## P1: manuscript and reporting corrections

### 7. Several numerical-section claims are unsupported or false

`reports/NUMERICAL_SECTION.tex` should not be used in its current form.

- Lines 46-51 claim an objective-reference error of order `1e-25` from the squared residual. This inference is unsupported and ignores the much larger independent-design transfer discrepancy.
- Lines 180-183 say that every iterative method reaches the reference on every logistic dataset and that terminal FR gradient norms lie between `1e-14` and `1e-8`. In the raw results, FR--R and FR--KL reach terminal gradient norms as large as about `2.5e-3`, particularly for `kappa_X=1`, and some terminal gaps are around `1e-6` to `9e-6`.
- Calling the four-point covariance-entry plot "bootstrap sharp" is stronger than the evidence. "Consistent with the predicted logarithmic dependence" is accurate.
- The explanation of Laplace predictive performance as under-dispersion is a causal interpretation not established by this experiment.
- The setup does not report the hardware, despite the planned experimental-setup requirements.
- Statements that the optimizer covariance is exactly the identity should be qualified as numerical construction/to numerical tolerance unless proved for the generated instance.

Recompute every quantitative statement directly from the final processed CSVs after the corrected rerun. Prefer restrained illustration language appropriate to a theory paper.

### 8. Documentation reports the wrong number of configs

The manuscript tree has 88 final YAML configs, not 31:

- 4 burn-in;
- 5 affine-equivariance;
- 1 anisotropic Gaussian;
- 12 global log-cosh;
- 6 local log-cosh;
- 60 logistic method/dataset configs.

These expand to 134 trajectories. Update `README.md`, `reports/REPRODUCIBILITY.md`, and `reports/MANUSCRIPT_PROTOCOL.md`. If configs are intentionally method-split, distinguish config count from trajectory count explicitly.

### 9. The manuscript audit and tests encode the implementation, not the release contract

The new tests pass, but they do not assert the most important manuscript invariants:

- independent update/evaluation/reference designs;
- one clean source hash and commit;
- zero failures and zero repairs;
- direct reference-residual margin;
- exact pilot/certified-step rule;
- successful smoke workflow.

Add these as regression tests or hard checks in `manuscript_audit.py`. An audit command that exits zero while listing a failed final trajectory is not a release gate.

### 10. Figure/data issues to fix during regeneration

- Figure 1(b): remove the invalid `10^8` failed/repaired comparison as described above.
- Figure 2: the grid panels are readable but crowded; preserve legibility at final manuscript width. The three local-rate radii almost overlap, which is acceptable if the caption states their purpose.
- Figure 3(c): the current terminal-gap summary is dominated by the constructed reference floor. Prefer iterations-to-tolerance or a certified terminal-gap summary after the reference fix.
- Align "wall-clock" wording with the actual measured timing scope if the code reports update-only algorithm time.
- Ensure each processed panel CSV contains the values actually plotted after masking/truncation, rather than relying on plotting code to reconstruct the displayed data.
- In `_grid_summary_table`, use the actual terminal gap rather than `nanmin(gaps)` when reporting a terminal value (`src/fr_gvi/plotting/manuscript_figures.py:503-529`).

## What is already in good shape

The following parts should be preserved:

- The main experiment suite is deterministic and organized around three experiment groups and three main figures.
- Final config method lists contain only FR--R, FR--KL, FB--GVI, and Laplace where applicable.
- No Wasserstein warm start, WFR hybrid, BWGD/BWSGD campaign, decreasing-stepsize campaign, lower-bound campaign, or affine-metric performance campaign appears in the main suite.
- The global log-cosh grid and local eigenmode experiment are appropriately reduced and use the intended target family.
- The local-rate implementation and its unit tests are structurally sound; the observed contractions are close to the predicted factors.
- The run system is resumable, raw failed runs are retained, config hashes agree with manifests, and panel provenance lists exact config IDs.
- Numerical code generally uses float64 and symmetric eigendecompositions for SPD matrix functions.
- Physical figure dimensions, font embedding, labels, and legends are broadly suitable.

Metadata dictionaries in some configs mention stochastic certified steps, but the executed final method lists do not run those methods. Filtering unused method metadata would reduce ambiguity.

## Required remediation order

1. Fix the smoke command and make all inexpensive validation gates pass.
2. Decide and document the final stepsize protocol before examining any new final results.
3. Separate logistic update, evaluation, and reference designs.
4. Repair the reference-selection/certification logic and make its checks hard failures.
5. Remove or stably reformulate the invalid affine `10^8` cell without covariance clipping or repair.
6. Add release-contract tests: no failures/repairs, clean unique provenance, independent designs, direct residual margin, and successful artifact provenance.
7. Commit the implementation and protocol with a clean tree; tag the frozen state.
8. Run the small pilot, commit `selected_steps.json`, and do not alter it after inspecting final results.
9. Run the reduced campaign once from the frozen revision.
10. Regenerate figures, processed CSVs, captions, tables, and provenance JSON.
11. Run `make test`, `make smoke`, `make manuscript-audit`, the plot audit, and a manual final-size visual audit.
12. Rewrite the numerical section from the corrected artifacts and independently verify every numerical claim.

## Acceptance criteria for manuscript readiness

The package is ready for a second audit only when all of the following are true:

- every declared final trajectory completes; zero final trajectories use clipping, covariance repair, backtracking, or undeclared rescue;
- all final manifests have one expected clean commit and code hash, and all config hashes match;
- `make test`, `make smoke`, `make manuscript-audit`, and the plot audit all exit successfully without scientific warnings;
- logistic update, evaluation, and reference designs are independent as specified and common across methods where appropriate;
- every non-Gaussian reference passes the stated residual margin on an independent finer design, with no material unexplained negative objective gaps;
- the stepsize selection is demonstrably frozen before the final campaign;
- figure provenance includes exact config IDs, commit, source hash, and clean-tree status;
- documentation gives correct config/trajectory counts and all manuscript claims match the final CSVs;
- the three figures remain readable at final manuscript size and do not present numerical failures or reference floors as convergence behavior.

Until these conditions hold, the current artifacts are useful exploratory diagnostics, not manuscript-ready numerical evidence.
