# experiment_L caption draft

Bayesian logistic regression with a proper Gaussian prior, $d\in\{10,50,100\}$, $n=10d$, $\lambda\in\{0.1,1\}$ and feature-covariance conditioning in $\{1,10^2,10^4\}$, using exact full-data derivatives. Each cell uses one fixed quadrature design shared by the updates, the objective evaluation and the reference, so deterministic gaps are exact for the discretized problem; the quadrature resolution floor of each cell is recorded in its reference manifest. (a) Deterministic methods, each at its own best swept step. (b) Stochastic methods at $B=16$, each run at a quarter of the largest step its own deterministic counterpart was measured to tolerate on that cell, so that no method is placed outside its stable range. (c) Best terminal gap across all 18 cells against the affine-invariant condition number. Laplace appears only as a non-iterative approximation-quality baseline.

Underlying data: `results/processed/experiment_L.csv`.
