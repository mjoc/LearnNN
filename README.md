# Neural Network Scratch Code #

* Some scratch work to study Bayesian Neural Networks (ala MacKay et al) and Networks with *Energy* Based Units
* First implement a regular FeedForward, Back Prop system
* Currently only set up for softmax output and tanh hidden units
* Back-prop is implemented for in-line, batch and mini-batch with stopping based on parameter number of epochs (no convergence checking yet)
* Some preprocessing of input (normalisation) is implemented but want to add eignevector stuff (PCA at least)
* Weight randomisation currently is only implemented as Gaussian with standard deviation as parameter
* Using GSL for matrix stuff for now
* Plan to create an R package for ease of use.
* Plan to use STAN for the Bayesian stuff, when I figure it all out



On the short term stack....

1. Regression Output unit
2. Other hidden unit types, maybe only ReLU
3. Momentum in Batch GD
4. Max Norm Weight Constraints (in unit input) in Batch GD
5. Dropout in learning
6. Convergence checking in weight fitting?
7. PCA (and other preprocessing options)
8. R package code, with graphical diagnostics (better do this sooner rather than later for testing)
9. Randomisation of input data with label-stratified sampling
