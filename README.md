# multiview-spectral-clustering-with-PALM
multiview spectral clustering solved with PALM in either gradient descent or closed-form solution

Task: Given multiview data G = {G^(1),G^(2), ... ,G^(v)}, cluster G into k clusters.

The objective function:
min ∑_a tr((G(a))^T L(a) G(a)) + λ∥G(a) −G∗∥_F^2 + θ∥AG∗∥_F^2    s.t. (G(a))^T G(a) = I
where L = D-W denotes the Laplace matrix, G* denotes the consensus matrix, and A represents the pair representation matrix of certain samples in the same cluster.

E.g. A = [1 -1 0 0 0 0 0 0 0 0; 0 0 0 1 -1 0 0 0 0 0; 0 0 1 0 0 -1 0 0 0 0; 0 0 0 0 0 0 1 -1 0 0; 0 0 0 0 0 0 0 0 1 -1].


This algorithm introduces PALM to solve the non-convex clustering problem.

Solution 1: G* is solved with gradient descent method;

Solution 2: G* is solved with closed-form method.


files:
PALM: PLAM solution;
SV: single view solution;
CSV: concate all features from different views.
