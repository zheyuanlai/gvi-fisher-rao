# Blocked experiments

## Experiment E: fixed-step bump-train sharpness

The manuscript states that "the sharpness of the resulting global rates and the
associated lower-bound constructions are deferred to the appendix", and that
appendix is not yet written. Searches of the full manuscript source and of the
downloaded references found no bump-train potential, smoothing rule, or
constants. Per the source-of-truth policy the construction was not invented.

Experiment E is therefore **not run**, by explicit decision rather than by
oversight. No surrogate target was substituted, and no figure claims a sharpness
result.

The generic infrastructure needed to add it later is already present: a
one-dimensional target interface, exact curvature constants, per-method certified
stepsize sweeps, strict failure recording, and manifest provenance. Supplying the
exact potential and constants would require adding one target class and one grid
builder; nothing else in the runner would change.

## Related optional constructions

The same reasoning applies to the other lower-bound and boundary constructions
named in the plan but absent from the manuscript:

- the smooth double-well boundary target;
- the convex-ridge local-gap target;
- the logarithmic-spiral continuous-time lower bound.

## What is covered instead

The sharpness question is not left entirely without evidence. The stepsize sweeps
in Experiments C, D and L report, for every cell and every method, the certified
step, the largest multiple of it that still makes progress, and the best terminal
gap at a fixed oracle budget (`results/tables/stepsize_summary.csv`). This
quantifies how conservative the certified steps are, which is the practical half
of the sharpness question, but it is not a lower-bound construction and is not
presented as one.
