# LowRankVortex.jl


This repository is a companion to the preprint Le Provost, Baptista, Marzouk, and Eldredge "A low-rank ensemble Kalman filter for elliptic observations", arXiv preprint (arXiv:2203.05120), 2022

In this paper, we introduce a regularization of the ensemble Kalman filter for elliptic observation operators. Inverse problems with elliptic observations are highly compressible: low-dimensional projections of the observation strongly inform a low-dimensional subspace of the state space. We introduce the *low-rank ensemble Kalman filter (LREnKF)* that successively identifies  the low-dimensional informative subspace where the inference occurs, performs the data-assimilation in tihs subspace and lifts the result to the original space.

This repository contains the source code and Jupyter notebooks to reproduce the numerical experiments in Le Provost et al. [^1]



## Installation

This package works on Julia `1.6` and above. To install from the REPL, type
e.g.,
```julia
] add https://github.com/mleprovost/LowRankVortex.jl.git
```

Then, in any version, type
```julia
julia> using LowRankVortex
```

We provide the routines to reproduce the numerical simulations and figures of the examples in
For examples, consult the documentation or see the example Jupyter notebooks in the Examples folder.


![](https://github.com/mleprovost/LowRankVortex.jl/raw/main/example2/setup_example2.png)

[^1]: Le Provost, Baptista, Marzouk, and Eldredge (2022) "A low-rank ensemble Kalman filter for elliptic observations," *arXiv preprint*, [arXiv:2203.05120](https://arxiv.org/abs/2203.05120).
