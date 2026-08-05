# Blocked experiments

The supplied manuscript source states that global sharpness and Gaussian local-region maximality are deferred to an appendix, but it contains no executable formulas for the requested one-dimensional bump train, logarithmic spiral, convex ridge, or smooth double well. Searches of the full 8,646-line source found no occurrence of those construction names and no matching potential definitions.

Per the source-of-truth policy, the repository does not invent these targets. Experiment E has an explicit skipped smoke manifest. The following remain blocked until exact formulas and constants are supplied:

- E: fixed-step bump-train sharpness;
- optional smooth double-well boundary behavior;
- optional convex-ridge local-gap scaling;
- optional logarithmic-spiral continuous-time lower bound.

Generic one-dimensional target and strict failure-recording infrastructure are already available, so exact formulas can be added without changing the runner.

