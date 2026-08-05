# Implementation notes

## Algorithms

`src/fr_gvi/algorithms/core.py` implements the six admitted iterative methods using the formulas in the task brief and manuscript. FR--R evaluates the covariance exponential through the symmetric eigendecomposition of `I - C^(1/2) A C^(1/2)`. FR--KL solves the precision-form system. FB--GVI follows Algorithm 1 of Diao et al.: the forward covariance is `M C M`, and the entropy JKO map is applied eigenvalue-wise.

The Price/Hessian--STL methods use the same Gaussian samples for the mean estimator and sampled Hessian. The S--FB--GVI baseline uses its native sampled gradient and Hessian. The raw-score switch exists only as an ablation flag and is not a main method.

## SPD policy

Matrices are symmetrized after floating-point products. SPD square roots, inverse square roots, logarithms, exponentials, and the FB--GVI JKO map use symmetric eigendecompositions. Cholesky solves replace dense inverses. A negative eigenvalue is repaired only when it is no smaller than `-100 * eps * max(spectral_scale, 1)`; the repair is placed in the trajectory. Larger violations raise `AlgorithmFailure` and are retained in the manifest.

## Expectation backends

- exact analytic expectations for Gaussian targets;
- dimension-wise Gauss--Hermite integration for affine images of separable log-cosh targets;
- fixed scrambled Sobol standard-normal designs for nonseparable deterministic updates;
- independent, larger Sobol designs for evaluation;
- iid Gaussian samples for stochastic updates.

All deterministic methods within one job share the same update engine. Evaluation designs have a separate seed.

## Reference optimizers

Gaussian references are exact. Log-cosh references solve the one-dimensional Gaussian first-order conditions at high Gauss--Hermite order and transform the solution affinely. Logistic references use a Cholesky parameterization, fixed scrambled Sobol samples, analytic reparameterization gradients, two starts, L-BFGS-B, and an independent doubled-size design for residual certification.

## Manuscript-specific mechanisms

The decreasing schedule is implemented exactly as `h_n = 8 kappa_star / (n+n0)`, `n0 >= 64 kappa_star^2`. The near-Gaussian local operator is assembled from the manuscript's exact `T`, `T*`, and `S` Gaussian score operators in optimizer-whitened coordinates.

