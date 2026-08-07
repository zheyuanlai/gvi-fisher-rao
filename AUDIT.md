# Second independent manuscript-readiness audit

Date: 2026-08-07

## Revised verdict

**Not yet manuscript-ready, but the numerical campaign is now close.**

The core rerun is substantially improved: all 131 declared final trajectories complete from one clean numerical-source hash, no final covariance repair is logged, the invalid affine (10^8) cell is gone, and all inexpensive gates pass. The remaining blockers are now mostly postprocessing, reproducibility, and protocol-consistency defects. They do not justify another full campaign unless the author rejects the three amended protocol choices listed below.

Do not cite the current Figure 3 or its predictive table until the pilot contamination and stale caption are corrected.

## Independent gate results

| Gate | Result |
| --- | --- |
| `make test` | Pass: 88 tests |
| `make smoke` | Pass |
| `make manuscript-audit` | Pass with no reported errors or warnings |
| Manuscript plot audit | Pass: 6.5-inch PDFs, matching PNGs, embedded fonts |
| Independent final-manifest audit | Pass: 87 configs, 131 manifests, all config/source/reference hashes and raw paths valid |
| Manual figure inspection | Figures 1 and 2 are usable; Figure 3 needs the corrections below |

The smoke run modified timestamp-bearing tracked artifacts. I restored only those audit-induced changes, and the working tree was clean afterward.

## What the revision genuinely fixed

The following first-audit blockers are closed:

- The final campaign now has 131/131 completed trajectories, one numerical source hash, one recorded campaign commit, and `git_dirty: false` throughout.
- The final method counts are 42 FR--R, 42 FR--KL, 32 FB--GVI, and 15 Laplace.
- No final trajectory records covariance repair, clipping, backtracking, or a rescue method.
- The affine grid stops at `cond(S)=10^6`; all twelve affine trajectories complete without repair. The failed and heavily repaired `10^8` observations are no longer plotted.
- `make smoke` is repaired.
- Figure provenance now includes a numerical source hash as well as a commit and clean-tree flag.
- The former finite-QMC logistic surrogate was replaced by deterministic one-dimensional quadrature over each linear predictor. This removes the old (10^{-3})-scale design-transfer defect.
- Reference suboptimality is now converted with the manuscript's proved gradient-domination inequalities rather than the former unjustified `0.5 * residual**2` heuristic.
- The false claim that all logistic FR terminal gradient norms were below (10^{-8}) was corrected.
- Documentation now uses 87 configs and 131 final trajectories in most places.

## P0: remaining release blockers

### 1. Figure 3's committed caption describes an obsolete experiment

`results/figures/manuscript/figure_3.md` and the caption generator in
`src/fr_gvi/plotting/manuscript_figures.py:885-920` still claim that:

- all methods use a shared 4,096-point scrambled-Sobol design;
- each iteration costs 246--250 ms;
- the expectation costs (O(Snd)).

None of these describes the final campaign. The current implementation uses order-48 and order-96 panelled one-dimensional quadrature, and the observed median update costs are approximately:

| Method | Median ms/iteration |
| --- | ---: |
| FB--GVI | 12.68 |
| FR--R | 13.08 |
| FR--KL | 13.95 |

The spread is about 10%, not 2%. The numerical-section draft has newer timing text, but the required caption artifact and its source are stale.

Required fix:

- update the Figure 3 lead and panel-(b) caption in the plotting source;
- regenerate Figure 3, its Markdown caption, processed CSVs, and provenance JSON;
- derive timing numbers programmatically from the final data instead of hard-coding them;
- make a regression test reject `Sobol`, `4096`, or the obsolete timings in the final logistic caption.

No trajectory rerun is needed.

### 2. The stepsize pilot leaks into Figure 3 and the predictive table

Pilot runs use experiment code `L` and tier `manuscript`, so
`load_experiment("L", "manuscript")` loads them together with the 60 final
logistic trajectories.

Consequences:

- `results/tables/manuscript_predictive.csv` reports six iterative datasets at
  (kappa_X=100), but only five Laplace datasets.
- `predictive_table()` in
  `src/fr_gvi/experiments/manuscript_tables.py:174-206` never filters pilot job IDs.
- Figure 3(c)'s provenance lists the three logistic pilot config IDs.
- The pilot changes the (kappa_X=100) resolution floor from about
  (1.73×10^{-13}) to (3.83×10^{-13}).
- The protocol says that pilot trajectories appear in no figure, which is therefore
  not true of the current artifact pipeline.

Required fix:

- give pilot results a separate tier/directory, or filter `job_id.startswith("pilot")`
  immediately in the common logistic loader;
- filter pilots explicitly in every manuscript table and panel;
- assert five datasets per method and condition;
- assert that no figure provenance `config_ids` entry starts with `pilot`;
- regenerate Figure 3 and every logistic summary table.

No final trajectory rerun is needed.

### 3. Figure 3(c) still presents constructed resolution floors as terminal gaps

`panel_logistic_conditions` clamps terminal gaps to a floor derived from negative
roundoff excursions and plots the clamped value as `terminal Delta(a_N)`.
The caption does not disclose the clamping, and markers at the floor are not
visually distinguished. This is the numerical-floor behavior the experiment brief
explicitly says not to present as a convergence effect.

Relevant code:

- `src/fr_gvi/plotting/manuscript_figures.py:757-772`
- `src/fr_gvi/plotting/manuscript_figures.py:808-865`

Required fix:

- use iterations to a declared relative tolerance across (kappa_X), as the
  original plan permits and the numerical-section draft already tabulates; or
- visually distinguish censored/floor observations and state exactly what was
  clamped.

The iterations-to-(10^{-6}) summary is the cleaner choice and requires no rerun.

### 4. The committed logistic pilot configs do not regenerate from the current code

I regenerated all manuscript configs into `/tmp` and compared them with the
committed tree. All 87 final configs reproduce exactly. The three committed
logistic pilot configs do not.

The current generator emits `quadrature_order: 48` and
`evaluation_quadrature_order: 96`; the committed pilot configs instead retain
obsolete `update_points/evaluation_points/reference_points: 4096` fields and
slightly different reference curvature and step scales. For example, the current
FR--R pilot scale is about `0.0062628654`, while the committed config and frozen
sweep record about `0.0062623135`.

The selected multipliers remain (2,2,1), so this has no material effect on the
final trajectories. It does mean the documented command
`make manuscript-pilot` mutates the supposedly frozen tagged protocol and does
not reproduce `selected_steps.json` exactly.

Required fix:

1. Regenerate the pilot configs from the current generator.
2. Rerun only the 42 pilot trajectories.
3. Regenerate `selected_steps.json`.
4. Verify that the multipliers remain (2,2,1).
5. Add a test that generates configs in a temporary directory and compares them
   byte-for-byte with the committed protocol.
6. Keep pilot configs outside the six final `configs/manuscript/` group
   directories, as the experiment brief requested.

If the multipliers remain unchanged and the 87 final configs remain byte-identical,
the 131 final trajectories do not need to be rerun.

### 5. The manuscript audit does not detect the defects above

The current audit correctly rejects failed/repaired final runs and multiple hashes,
but it does not check:

- committed configs against the generator;
- manifest hashes against the current source and config files;
- pilot IDs in figure/table inputs;
- five logistic datasets per method and condition;
- caption text against the configured expectation backend;
- the exact commit/tag relationship.

My independent check found no mismatch among the 131 final manifests, but these
invariants need to be automated before release.

## Author decisions required: three intentional protocol amendments

These are not hidden bugs. Claude documented deliberate changes to the requested
protocol. They are scientifically defensible in parts, but Claude should not treat
them as approved merely because an audit command passes.

### A. Stepsize selection

Requested:

- one pilot target;
- `h = eta * h_cert`;
- largest admissible multiplier.

Implemented:

- one pilot per target family;
- `h = eta / [beta_star * max(lambda0_star_max, 1)]`;
- fastest admissible multiplier, minimized across the two families;
- a separate theorem ceiling for FB--GVI.

This remains a direct deviation from the brief and was motivated after inspecting
behavior on final-grid cells. Either the author must explicitly approve this as the
new protocol, or Claude must restore the requested rule and rerun the affected
non-Gaussian experiments. Relabeling the amendment as preregistered is not enough.

### B. Logistic expectation rule

Requested: common deterministic QMC update design, independent finer evaluation
design, and a still finer reference design.

Implemented: deterministic panelled Gauss--Legendre integration of the exact
one-dimensional predictor marginals.

The implemented rule is scientifically preferable for this target, but it is
quadrature, not QMC and not an independent random design. My independent comparison
on all 15 final datasets at initial, intermediate, and reference states found:

- order 48 versus order 160: maximum objective/gradient/Hessian differences about
  (6.6×10^{-11}), (6.5×10^{-12}), and (2.9×10^{-11});
- order 96 versus order 192: maximum absolute objective difference about
  (1.3×10^{-10}).

This is adequate for the experiment but does not support saying that every
expectation is literally exact or that the rules agree uniformly to (10^{-11}).
If approved, describe the backend as deterministic one-dimensional quadrature to
reported numerical precision, use the configured quadrature-order fields rather
than hard-coded values, and add a full-grid fine-rule certification.

### C. Reference acceptance criterion

Requested: Fisher--Rao residual at least two orders below the smallest reported
objective gap.

Implemented: convert the squared residual to a certified gap with a proved PL or
gradient-domination inequality, then compare like units.

The implemented comparison is mathematically better dimensioned, but it is not the
literal requested test. Also, the numerical section promotes a Bures--Wasserstein
residual to a reference-free certificate at every iterate, which risks making
Wasserstein geometry part of the proposed numerical methodology rather than merely
an external FB--GVI baseline.

A Fisher--Rao-only certificate is sufficient: using the squared FR residual divided
by `alpha_star * lambda_min(C_star)`, the worst reference still has about
6.86 decades of margin below the smallest plotted gap. Prefer that branch in the
numerical audit and text. No trajectory rerun is needed.

If the author insists on the literal residual-versus-gap rule, the plots must stop
at a higher gap; the present residuals are around (10^{-13}), while some plotted
gaps reach (10^{-14})--(10^{-16}).

## P1: manuscript-facing corrections

### 6. Several statements remain inaccurate or internally inconsistent

- `reports/NUMERICAL_SECTION.tex:35-47` and
  `reports/MANUSCRIPT_PROTOCOL.md:115-133` call finite quadrature "exact" and
  overstate its measured cross-order agreement.
- `reports/NUMERICAL_SECTION.tex:231-235` says the three iterative predictive NLLs
  agree to four decimals on every dataset. They do on 13 of 15 datasets; the maximum
  within-dataset difference is (1.85×10^{-5}), and two datasets round
  differently at four decimals. "Agree within (1.9×10^{-5})" is accurate.
- `reports/MANUSCRIPT_PROTOCOL.md:235-249` still describes the obsolete shared-QMC
  evaluation protocol, contradicting its own exact-quadrature section.
- `reports/MANUSCRIPT_PROTOCOL.md:159-164` still claims a 2% per-iteration timing
  spread.
- `scripts/run_manuscript.sh:2` still says 134 trajectories.
- `README.md:21` says "pilot cell" although there are two.
- `README.md:116-120` still advertises five old manuscript composites rather than
  the current three figures.
- `reports/REPRODUCIBILITY.md:148-153` claims CSVs reproduce bit-for-bit, but the
  CSVs contain wall-clock timings. Restrict the claim to deterministic scientific
  columns.
- The campaign manifests record commit `5e6b2416`, while
  `numerics-protocol-v2` points to later commit `01a245e3`. The tagged tree has
  the same numerical source hash and final configs, so reproduction is possible,
  but "the campaign was run from the tagged revision" is not literally true.
- Use the published/requested method name `FB--GVI` in figures and text rather
  than inventing `BW-FB`; the latter needlessly obscures which Diao et al.
  algorithm was run.

### 7. Processed summaries should not contain misleading terminal values

`_grid_summary_table` still records
`terminal_normalized_gap = nanmin(gaps)`, which is often a negative roundoff
excursion rather than the actual terminal value. It is currently not used because
all global cells reach tolerance, but the processed CSV is part of the manuscript
artifact. Record the actual terminal value or omit the column.

## Manual figure assessment

- **Figure 1:** readable and scientifically coherent after removing the (10^8)
  cell. The shaded non-representable region is helpful. Rename the baseline
  FB--GVI.
- **Figure 2:** readable at manuscript width. Six panels are dense but coherent.
  The local-rate panel communicates the theorem well.
- **Figure 3:** visually clean, but panel (c) should become iterations-to-tolerance.
  The committed caption is invalid, and pilot leakage affects the (kappa_X=100)
  floor and predictive table.

## Recommended remediation order

1. Obtain the author's explicit decision on amendments A--C.
2. Separate pilot outputs from final manuscript data and fix all pilot filters.
3. Regenerate the three logistic pilot configs and pilot sweep only.
4. Confirm the frozen multipliers and final config files do not change.
5. Replace Figure 3(c) with iterations-to-tolerance.
6. Correct the Figure 3 caption generator, method label, timings, protocol note,
   predictive table, and stale counts.
7. Use the Fisher--Rao-only reference certificate in manuscript-facing reporting.
8. Add release tests for config regeneration, pilot exclusion, dataset counts,
   caption backend, current hashes, and tag/provenance consistency.
9. Regenerate figures and tables only.
10. Rerun `make test`, `make smoke`, `make manuscript-audit`, and the plot
    audit, then perform one final visual/text audit.

## Acceptance criteria for a third audit

The package is ready for a final audit when:

- all three intentional protocol amendments are explicitly accepted or reverted;
- committed pilot and final configs reproduce byte-for-byte from the generator;
- no pilot ID or pilot observation enters a main figure, table, or provenance list;
- every logistic aggregate contains exactly five datasets;
- Figure 3's caption matches the actual quadrature backend and final timings;
- Figure 3(c) does not portray a constructed floor as a measured terminal gap;
- manuscript-facing reference certification uses the approved Fisher--Rao criterion;
- all stale counts, QMC descriptions, timing claims, and predictive claims are fixed;
- all gates pass from a clean tree.

The current 131 final trajectories are credible and likely reusable. The present
blockers do not require another full campaign unless the author chooses to restore
the original QMC or certified-step pilot protocols.

---
