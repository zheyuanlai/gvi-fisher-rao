# figure_4 caption draft

Gaussian variational inference for Bayesian logistic regression on five real binary-classification datasets, with a proper Gaussian prior $\theta\sim\mathcal N(0,I)$. Features are standardized with training-split statistics and an intercept is appended, so every problem is strongly log-concave and smooth with constants available in closed form. Objective gaps are measured against an independently solved reference certified far below the smallest gap drawn.

**(a)** Objective gap against the time spent inside the update, on ionosphere. Every stepsize comes from the implementable dyadic rule: a base scale computed from the geometry of the update and one expected Hessian at the initialization, then a multiplier selected on a disjoint subsample of the training rows by lowest training objective. No optimizer-whitened constant and no reference solution enters the choice.
**(b)** The same problem for the stochastic arms at a common batch size, median over seeds with a 10-90 per cent band. Reading (a) and (b) together separates the two axes of the comparison: Price-BBVI shares its second-order estimator with S-BW-FB and its parameter space with BBVI-STL, so the pattern of agreement identifies which of the two the behaviour follows.
**(c)** Held-out predictive negative log-likelihood against algorithm time. The curves converge to a common level, as they must: every iterative method here targets the same Gaussian variational optimum, and the Laplace reference targets a different Gaussian. What the panel measures is how quickly a usable posterior approximation is reached, which is the quantity a practitioner spends compute on.
**(d)** Algorithm time to a relative objective gap of $0.0001$ on each dataset, median over seeds for the stochastic arms; markers are offset horizontally for legibility. Averaged over the five datasets the best mean ranks are FR-KL-STL 1.2, Price-BBVI 2.2, BBVI-STL 2.7. Absent markers did not reach the tolerance inside the horizon: BBVI-STL on ionosphere, BBVI-STL on sonar, BW-FB on sonar, Price-BBVI on sonar, S-BW-FB on sonar. No method is uniformly fastest, which is the expected outcome once every method is tuned by the same implementable rule: affine invariance is a structural property, not a claim of numerical dominance.

Panel data:
- (a) `results/processed/manuscript/figure_4_a.csv`
- (b) `results/processed/manuscript/figure_4_b.csv`
- (c) `results/processed/manuscript/figure_4_c.csv`
- (d) `results/processed/manuscript/figure_4_d.csv`
