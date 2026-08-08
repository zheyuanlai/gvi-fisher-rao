# Manuscript numerical protocol

The exploratory campaign in `configs/full` sweeps stepsizes and replicates; it
exists to locate stable operating points and to check the implementation against
the theory. The manuscript needs far less. This note records the reduced,
preregistered protocol behind the numerical section, the decisions it rests on,
and the places where the executed protocol differs from the plan it was written
against.

## Scope

Eleven experiment groups, four main-text figures, three appendix figures, one
accounting table, 154 config files, 1821 trajectories.  The config count exceeds
the cell count because the logistic and benchmark groups are split one file per
(cell, method) so the campaign, which parallelizes over config files, can run
those trajectories independently.

| Group | Figure | Trajectories | Purpose |
|---|---|---|---|
| Gaussian structure and affine invariance | 1 | 8 + 12 + 3 = 23 | the central geometric motivation |
| Global-to-local deterministic convergence | 2 | 36 + 12 = 48 | the global and local theorems on non-Gaussian targets |
| Stochastic mechanisms | 3 | 150 + 1050 + 240 = 1440 | Section 4: cancellation, and the two factors of the `Delta t / B` floor, and the rescue |
| Practical benchmark on real posteriors | 4 | 150 | the methods on data someone actually has |
| Synthetic logistic conditioning sweep | A3 | 60 | the controlled version of the benchmark |
| Dimensional scaling and oracle cost | A2 | 100 | where the dense full-covariance cost goes |

The stochastic experiments were previously left outside the manuscript, on the
grounds that Section 4 is a set of guarantees rather than a benchmarking
contribution.  That was the wrong call.  An entire section of algorithmic theory
with no numerical demonstration is a gap, and the mechanisms it predicts --
pathwise cancellation and a stationary floor proportional to `Delta t / B` -- are
exactly the kind a measurement can falsify.  They are now Figure 3.

An earlier version of this campaign also measured the `O(1/N)` tail of a
decreasing-stepsize schedule.  That theorem was in an earlier draft of the
manuscript and is not in the current one, so the panel validated a result the
paper does not state; it was replaced by the stepsize sweep, which measures the
`Delta t` factor of a floor the paper does state and which nothing else in the
campaign checked.

## Methods

Nine iterating methods plus `Laplace`, on two axes:

| | Fisher--Rao | Bures--Wasserstein | Parameter space |
|---|---|---|---|
| Deterministic | `FR--R`, `FR--KL` | `FB--GVI` | `Sq--NGVI` |
| Price/Hessian estimator | `FR--R--STL`, `FR--KL--STL` | `S--FB--GVI` | `Price--BBVI` |
| Gradient-only estimator | -- | -- | `BBVI--STL` |

`Price--BBVI` shares its estimator with `S--FB--GVI` and its parameter space with
`BBVI--STL`, which is what turns a one-way geometry comparison into a two-way
one.  Still excluded: no Wasserstein warm start, no Wasserstein--Fisher--Rao
hybrid, no BWGD or BW--SGD, no projected Fisher--Rao variant, no Euclidean or
quasi-Newton optimizer, no diagonal or low-rank family.  `make manuscript-audit`
fails if any run outside the admitted set appears.

The three external comparators are implemented from the papers' own algorithm
statements: `Sq--NGVI` is Kumar et al. Algorithm 1 and Equation (14),
`Price--BBVI` is Kim et al.'s SPGD in the parameterization of their Assumption
2.2 with the Bonnet--Price estimator and the closed-form entropy prox, and
`BBVI--STL` is Kim et al.'s projected SGD with the Definition 5 estimator and the
Proposition 1 projection.  Their safeguards are applied because they are part of
the algorithms and of the domains their guarantees are stated on; omitting one
would not be a fairer comparison, it would be a comparison against an algorithm
nobody proposed.  Every activation is counted and reported.  None of our methods
has a safeguard, and `Sq--NGVI` has none either, so its factor breakdowns are
recorded as failures.

## Stepsize protocol

The final figures show no stepsize sweep.  Three regimes coexist and every
method specification declares in its config which one it is in, so each is
checked by a different gate.

| Rule | Where | What it reads |
|---|---|---|
| `certified` | panels that verify a theorem | the step that theorem admits |
| `capped_practical` | the `1/B` floor panel | deliberately above the certified step, see below |
| `pilot_frozen` | the deterministic non-Gaussian comparisons | an optimizer-whitened scale, see below |
| `implementable_grid` | the whole benchmark, the cancellation panel, the scaling study | nothing the theory reserves for the analysis |

The `capped_practical` departure is deliberate and is the only place a panel runs
above its certified step: the certified step scales like `1/kappa_star`, and at
that step the horizon needed to leave the deterministic transient exceeds any
affordable budget, so a measured floor would be flat in `B` for a reason that has
nothing to do with the theorem being tested.

### The implementable rule

`pilot_frozen` is written in `beta_star` and `lambda_{0,star}^max`, which require
`C_star`.  That is fine for a diagnostic, whose job is to place every method at a
comparable point of its own stable range, and indefensible for a benchmark.  The
benchmark instead uses:

- **base scale** `h_0 = 1` for the affine-equivariant natural-gradient updates,
  whose stepsize is dimensionless, and `h_0 = 1/lambda_max(H(a_0))` for the
  Euclidean ones -- one expected Hessian at the initialization;
- **candidates** `h_0 2^{-k}`, `k = -3 .. 11`;
- **admissibility** positive definite covariance, finite objectives, no repair
  logged, objective decrease over a fixed pilot horizon;
- **selection** the lowest *training* objective among the admissible.

Selection runs on a disjoint 50% subsample of the training rows, so no number a
figure reports was used to choose a stepsize, and the base scale is recomputed on
the full problem.  The tuning cost is counted in oracle pairs and reported.

Two refinements were forced by the data, not added for symmetry.

1. **Three problem instances, most conservative choice.**  A multiplier selected
   on one draw of a randomly generated cell diverged on the next: on the Gaussian
   cancellation cell the stability boundary moves by a full grid factor between
   draws.  Fixed datasets have one instance and are unaffected.
2. **Calibrate at the deployment batch size.**  A stochastic step admissible at
   `B = 16` is not admissible at `B = 1`; the noise the update absorbs scales like
   `1/B`.  Tuning the cancellation panel at the benchmark's batch size selected a
   step that diverged at the panel's own.

Both were caught by an explicit transfer check -- every selected step rerun at the
full horizon on the instance the campaign will use -- before the campaign was
launched.  That check is the reason the campaign has no failed trajectories.

### The frozen practical multipliers

**Pilots.** Two cells, one per target family, because a single cell cannot
calibrate both regimes: the whitened log-cosh cell starts with its covariance
already in the band, the logistic cell starts two orders of magnitude above it.

| Family | Cell | Master seed |
|---|---|---|
| optimizer-whitened log-cosh | `(d, kappa_base, rho) = (10, 10, 1)` | `20260401` |
| Bayesian logistic regression | `(d, kappa_X) = (50, 10^2)` | `20260402` |

Each has the nominal parameters of a cell that appears in the figures but a
*different master seed*, hence a different target instance, so no trajectory shown
in any panel was used to select a stepsize.

**Scale.** The multiplier applies to `1/(beta_star * max(lambda_{0,star}^max, 1))`.
What has to stay of order one is `h beta_star` times the whitened covariance, and
the whitened covariance is largest at the initialization whenever `C_0` overshoots
`C_star`. See the first deviation below for why neither the certified step nor
`1/beta_star` alone transfers.

**Admissibility.** A multiplier is admissible when the covariance stays positive
definite for the whole horizon, every recorded objective is finite, the objective
decreases over the horizon, and no clipping, repair or backtracking is logged.

**Selection.** Among the admissible multipliers, the fastest: the one reaching a
relative gap of `1e-6` in the fewest iterations, ties broken towards the smaller
step, then the minimum across the two pilot families so one number is admissible on
both. `FB--GVI` is additionally held at multiplier one, where Diao et al.'s
requirement `eta <= 1/beta` ends.

Result, in `configs/manuscript/selected_steps.json`:

| Method | Frozen multiplier | Log-cosh boundary | Logistic boundary |
|---|---|---|---|
| `FR--R` | 2 | 4 | above 16 |
| `FR--KL` | 2 | 4 | above 16 |
| `FB--GVI` | 1 | 4 | 16 |

The log-cosh family binds. Certified steps are recorded per cell in every config
and reported in `results/tables/manuscript_stepsizes.csv`, which also gives the
ratio of the step used to the step certified:

| Experiment | Method | Step / certified step |
|---|---|---|
| log-cosh grid | `FR--R` | 4.2 to 194 |
| log-cosh grid | `FR--KL` | 2.1 to 97 |
| log-cosh grid | `FB--GVI` | 1 |
| logistic | `FR--R`, `FR--KL` | 4 |
| logistic | `FB--GVI` | 1 |

The spread in the Fisher--Rao rows is the mechanism behind deviation 1 below: the
certificate's conservatism tracks `kappa_star`, which ranges over two decades on
the grid, while on the logistic cells it is a clean factor of four. `FB--GVI` runs
at exactly its own certified step everywhere.

## Initialization

| Experiment | Initialization |
|---|---|
| Gaussian burn-in | `m_0 = 0`, `C_0 = lambda_0 I` |
| Affine equivariance | base state carried through `x -> Sx + b`, `cond(S)` up to `10^6` |
| Anisotropic Gaussian | `m_0 = 0`, `C_0 = I` |
| Global non-Gaussian | `m_0 = 2 e_1`, `C_0 = I`, in optimizer-whitened coordinates |
| Local rate | `a_0 = a_star + r v_min` along the slowest eigenmode of `L_star` |
| Logistic regression | `m_0 = 0`, `C_0 = lambda_prior^{-1} I` |
| Stochastic cancellation | `C_0 = C_star` matched, `m_0` displaced |
| Floor and stepsize sweep | quadratic rescue, then the stochastic iteration |
| Real-data benchmark | `m_0 = 0`, `C_0 = lambda_prior^{-1} I` |

The non-Gaussian targets are constructed so that `C_star = I` exactly: the
separable optimizer `diag(sigma^2)` does not depend on the affine map, so
`T = Q diag(1/sigma)` gives `C_star = T diag(sigma^2) T^T = I`. Starting at
`C_0 = I` then puts the covariance inside the band from the first iteration, and
the trajectory measures non-Gaussian localization rather than a covariance
burn-in. The whitened curvature constants are unchanged by the relabelling.

## Reference solutions

Every expectation in the manuscript campaign is exact, so no reported gap is a gap
to the minimizer of a sampled surrogate.

| Family | Rule |
|---|---|
| Gaussian | closed form |
| log-cosh | panelled Gauss--Legendre per coordinate, order 32 for updates and 64 for evaluation |
| logistic | panelled Gauss--Legendre per linear predictor, order 48 for updates and 96 for evaluation and the reference |
| real-data logistic | the same rule; the reduction to one-dimensional integrals is a property of the likelihood, not of the design |

The logistic reduction is what makes this possible: `V` depends on `theta` only
through the linear predictors `z_i = x_i . theta`, and `z_i` is a scalar Gaussian
under `q`, so the objective, the expected gradient and the expected Hessian are `n`
independent one-dimensional integrals. The evaluation rule is strictly finer than
the update rule and the two agree to about `1e-11`, which is the "independent finer
evaluation design" the plan asks for, with a transfer error twelve orders below the
gaps the figures resolve rather than the `1e-3` a 4096-point quasi-Monte-Carlo
design leaves.

A residual is a gradient norm and a gap is an energy difference, so the audit
converts between them with the manuscript's own Gaussian variational PL inequality,

    |g|^2 + Tr((C^{-1}-H) C (C^{-1}-H)) >= 2 alpha_star Delta,

whose left side is the Bures--Wasserstein residual. This is a proved bound with an
explicit constant the campaign computes exactly, so the reference's own
suboptimality is certified rather than inferred. `make manuscript-audit` requires
that certified bound to sit at least two orders of magnitude below the smallest gap
any figure resolves, and fails on any objective gap that goes materially negative,
since a trajectory passing below the reference is direct evidence that the
reference is not the minimizer.

The same two lemmas give a **reference-free certificate** at every iterate,
`Delta(a) <= min(BW^2/(2 alpha_star), FR^2/(alpha_star lambda_min(C)))`, recorded
as `certified_gap` in every trajectory. It holds at every state where the measured
gap is meaningful and is tight to a factor of `1.2` on the log-cosh grid.

## Reported quantities

`Delta(a_n)/Delta(a_0)`, the Fisher--Rao gradient norm `||grad E(a_n)||_{a_n}`, and
`||a_n - a_star||_star`, against iterations, elapsed flow time `nh`, and wall-clock
time. Each deterministic iteration consumes one expected gradient and one expected
Hessian, so the iteration count is already a population-oracle count. Wall-clock is
measured inside the update alone, excluding the per-iteration diagnostics. On these
problems it adds little to the iteration count: the three methods cost `12.2` to
`14.2` milliseconds per iteration, a spread of about sixteen per cent, because the
expectation dominates the `O(d^3)` linear algebra that separates the matrix
exponential of `FR--R` from the resolvent solve of `FR--KL`. The figure captions
compute these numbers from the final data rather than quoting them.

Curves are truncated at the resolution floor of the quantity being plotted, so a
numerical plateau is never drawn as if it were convergence.

## Workflow

```bash
make test
make manuscript-datasets  # fetch and hash-pin the five real datasets, once
make manuscript-pilot     # sweeps the pilot and writes selected_steps.json
make manuscript-tuning    # implementable steps for the benchmark and scaling study
make manuscript-configs   # instantiates the 152 configs from both step records
make manuscript-runs      # 1821 trajectories
make manuscript-figures   # figures 1-4 and a1-a3, plus the width and font audit
make manuscript-tables
make manuscript-audit
```

Do not change the selected steps after inspecting final results. Every figure
carries a provenance JSON naming its panels, their config identifiers and the
commit they were produced from; every panel carries its processed CSV.

## Deviations from the plan this protocol was written against

All eight are recorded in `configs/manuscript/protocol_amendments.json` with what
was requested, what was implemented, the reason, the cost, and the author's
decision.  A--D were approved on 2026-08-07 and E--H on 2026-08-08. `make manuscript-audit` fails if
an amendment is undecided, records no decision maker, omits its cost, or if the
implementation drifts from the fingerprint that was approved, so a deviation
cannot pass merely because no gate mentions it.

The first four were each forced by something the data showed; E--H are the
scope changes that turned a deterministic theorem-diagnostic section into one
that also demonstrates Section 4 and benchmarks against current methods.  Their
full statements are in the JSON; in brief:

- **E** adds `Sq--NGVI`, `Price--BBVI` and `BBVI--STL`, each with its own
  published safeguard, counted.
- **F** replaces the optimizer-whitened practical scale with the implementable
  rule, for the benchmark and the scaling study only.
- **G** replaces synthetic logistic designs with five hash-pinned real posteriors
  in the headline comparison, retaining the synthetic sweep in the appendix.
- **H** compresses the main text to four figures and one table and moves the rest
  to an online appendix.


### 1. The frozen multiplier applies to a different scale, on two pilot cells

As specified, one pilot selects a multiplier of each method's *certified* step and
freezes it for every non-Gaussian experiment. Executed literally, this diverges on
six of the twelve final grid cells. Two intermediate scales were tried and each
failed on one family; the executed protocol uses the third.

What has to stay of order one for any of the three schemes to be stable is
`h beta_star lambda_max(C_star^{-1/2} C_n C_star^{-1/2})`: the Riemannian
retraction exponentiates `h (I - R H R)`, and `R H R` is bounded by `beta_star`
times the whitened covariance, which is largest at the initialization whenever
`C_0` overshoots `C_star`.

| Scale | Fails because |
|---|---|
| certified step `1/(2 beta_star lambda_max_star)` | carries the growth allowance `lambda_max_star >= 1/alpha_star`, never attained on the whitened log-cosh cells, so its conservatism grows like `kappa_star`: a multiplier chosen at `kappa_star = 11` diverges at `kappa_star = 2` |
| `1/beta_star` | ignores the initialization and diverges on the logistic posteriors, where `C_0 = lambda_prior^{-1} I` overshoots `C_star` by two orders of magnitude |
| `1/(beta_star * max(lambda_{0,star}^max, 1))` | used; the certified step with the non-attained growth allowance removed and the initialization kept |

On the certified scale the pilot's Fisher--Rao sweep is censored at the top of the
prescribed multiplier grid, so no boundary is located. On the scale used, the
boundary is located for every method on both families, and one frozen number
transfers.

A single pilot cell also cannot calibrate both families, because the whitened
log-cosh cell has `lambda_{0,star}^max = 1` exactly while the logistic cell has it
near `10^2`; the protocol therefore uses one pilot per family and takes the
smaller multiplier.

The certified steps are unchanged, still recorded per cell, and still used
directly by the panels that verify a theorem.

### 2. A speed criterion was added to the selection

The four admissibility criteria end at the *stability* boundary, which is not the
fastest step. A Fisher--Rao step with `h gamma_star` near two is admissible by all
four criteria yet converges slowly, because the linearized one-step factor
`1 - h gamma_star` approaches `-1`. The selection therefore ranks the admissible
multipliers by iterations to a `1e-6` relative gap on the pilot. On the log-cosh
pilot this picks multiplier two for both Fisher--Rao schemes and one for `FB--GVI`,
whose multiplier two does not reach the tolerance at all.

The residual effect of the boundary is still visible and is reported: at
`kappa_star` near `1.1` the frozen step lands closest to `h gamma_star = 2` and the
Fisher--Rao iteration count rises. That is panel 2(d).

### 3. The logistic expectations are deterministic quadrature, not a sampling design

The plan asks for a common quasi-Monte-Carlo update design, an independent finer
evaluation design and a still finer reference design. That structure exists to
control a sampling error the logistic problem does not have to incur: `V` sees
`theta` only through the linear predictors, so all the expectations are
one-dimensional integrals and are computed by a panelled Gauss--Legendre rule
instead.

The plan's intent is met and its numbers are not. The evaluation and reference
rules (order 96) are distinct from and strictly finer than the update rule (order
48), and all methods share both. What changes is the size of the residual
disagreement: against an order-160 rule the order-48 objective, gradient and
Hessian differ by at most `6.6e-11`, `6.5e-12` and `2.9e-11`, where a 4096-point
scrambled-Sobol design misplaced the objective by about `1e-1` and left a
design-transfer error near `1e-3`. A gap reported here is a gap to the Gaussian
variational optimum, not to the minimizer of a sampled surrogate.

### 4. Reference certification converts residuals with a proved inequality

The plan asks for the Fisher--Rao residual to sit two orders of magnitude below
the smallest reported gap. A residual is a gradient norm and a gap is an energy
difference, so the audit instead converts one into the other with Fisher--Rao
gradient domination and compares like with like. The worst reference then carries
`6.9` decades of margin. Applying the literal residual-against-gap test would
require the panels to stop at a much larger gap, since the residuals are near
`1e-13` while the plotted gaps reach `1e-16`.

## Correctness fixes made along the way

Two latent bugs surfaced while building the reduced campaign.

- `ShiftedLogCoshTarget` tested invertibility with `|det T| < eps`. That is a test
  on scale, not on conditioning: at `d = 50` a map with `cond(T) = 7` has
  `|det T| ~ 1e-22` and was rejected. The test is now on the singular-value spread.
- The log-cosh marginal solve integrated the stationarity conditions on a single
  Gauss--Legendre panel whose nodes scale with the marginal width, while the
  expectation engine uses a panelled rule for exactly the reason that fails: in the
  small-curvature coordinates the marginal is several units wide but the `tanh` and
  `sech^2` structure lives in a window of unit width. The reference residual on the
  wide-marginal cells was `8e-8` against `3e-13` elsewhere. The solve now uses the
  same panelled rule and the residual is uniform.
