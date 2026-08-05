# experiment_I caption draft

Estimator ablation and the sticking-the-landing variance bound, on a Gaussian and a shifted log-cosh target with $d=8$ and $B\in\{1,4,16\}$, along a path interpolating from the initialization to the optimizer. (a) The intrinsic Fisher--Rao variance of the STL mean estimator relative to the raw-score estimator collapses as the state approaches $a_\star$; for the Gaussian target it is exactly zero at $a_\star$, which is off a logarithmic axis and so is omitted from the panel. (b) The measured Fisher--Rao tangent variance (thin lines with markers) against the Lemma 4.7 bound $(2\|\mathrm{grad}\,\mathcal E\|_a^2 + \tfrac32\Psi(a))/B$ (thick pale lines). (c) The bound holds at every state and batch size, with a worst-case ratio of 0.870. This is an estimator ablation, not one of the six compared algorithms.

Underlying data: `results/processed/experiment_I.csv`.
