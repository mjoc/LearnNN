# Neural Network (simple FF MLP) Scratch Code #
## This code is not a working version as of yet, but may work ok

* Some scratch work to eventually study Bayesian Neural Networks (ala MacKay et al) and Networks with *Energy* Based Units
* First implement a regular FeedForward, Back Prop system
* Currently only set up for softmax output and tanh hidden units
* Back-prop is implemented for in-line, batch and mini-batch with stopping based on parameter number of epochs (no convergence checking yet)
* Nestorov Momentum and some other stuff is implemented
* Using GSL for matrix stuff for now
* Test script can be found as a gist named 'antfarm_test1.R' (https://gist.github.com/mjoc/)
* Very interactive with R (via Rcpp) to study graphical and statistically things like convergence and other properties such as internal representation 

On the short term stack....

1. Make code work in R without GSL installed, maybe use rcppEgigen or Armadillo
2. Max Norm Weight Constraints (in unit input) in Batch GD
3. Dropout in learning
4. Convergence checking in weight fitting?
5. PCA is implemented but needs to be redone with SVD for numerical safety
6. Other hidden unit types, maybe only ReLU
7. More graphical diagnostics, on the R side
8. Bayesian fitting, using Stan if possible, and also other non-Bayesian schemes like (direct) feedback alignment for comparison. Are there shared libraries for this
9. Put in collecting feedforward values as an option (currently always on) and other implementation optimisations to deal with larger networks
