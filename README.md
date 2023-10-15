# Implementation for the paper: "Bayesian Optimization under Stochastic Delayed Feedback"

This directory contains the code for the SVM experiment in Section 6.2. It includes implementations of all methods under comparison: GP-UCB-SDF (ours), GP-TS-SDF (ours), GP-UCB, asy-TS, GP-BUCB, GP-BTS.

## Dependencies:
- GPy (https://github.com/SheffieldML/GPy)
- scikit-learn (for running SVM)
- scipy, numpy, pandas

## Scripts:
- run_svm_delayed.py: runs the different algorithms.
- bayesian_optimization_delayed.py: implementations of different methods
- aux_funcs_delayed.py: some auxiliary functions
