# figure_a2 caption draft

Online appendix: where the dense full-covariance cost goes as the dimension grows. Synthetic Bayesian logistic regression with $n=10d$ and a proper Gaussian prior, five problem instances per dimension, every method at a stepsize chosen by the same implementable rule as the practical benchmark. Timings are within-machine comparisons on one host with BLAS pinned to a single thread.

**(a)** Dense linear-algebra time per iteration against dimension, with the model oracle drawn separately in grey and a $d^3$ reference. At $d=10$ the oracle dominates every method's algebra by a factor of 10, which is why the main-text wall-clock panels largely repeat the shape of their iteration-count panels. By $d=200$ the algebra has overtaken the oracle for none. This is where the full-covariance assumption starts to cost: the storage is $O(d^2)$ and the per-iteration algebra is $O(d^3)$ for every method here, and the schemes separate by their constants - the matrix exponential of the retraction against the resolvent solve of the Bregman scheme against the single Cholesky update of the square-root method.
**(b)** Objective gap after a common iteration budget, median over problem instances. Read with (a) this answers whether a method's iteration count survives the cost of producing it: a scheme that needs fewer iterations but pays more per iteration only wins where the gap between the curves in (a) is smaller than the gap between the curves here.

Panel data:
- (a) `results/processed/manuscript/figure_a2_a.csv`
- (b) `results/processed/manuscript/figure_a2_b.csv`
