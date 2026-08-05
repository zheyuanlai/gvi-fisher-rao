# experiment_H caption draft

Sticking-the-landing cancellation on a Gaussian target initialized with the matched covariance $C_0=C_\star$ and a mismatched mean, $B=1$, 30 paired seeds, $d\in\{2,10,50\}$. The Fisher-Rao STL mean noise is proportional to $C_n-C_\star$ and so vanishes pathwise once the covariance is matched (Corollary 4.20): the two Fisher-Rao schemes produce bit-identical trajectories across all 30 seeds at 55\% of recorded iterations, and never differ by more than floating-point roundoff. S-FB-GVI retains its native estimator noise at $O(1)$. This compares complete algorithms together with their native estimators, not geometry alone.

Underlying data: `results/processed/experiment_H.csv`.
