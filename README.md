# GPs-for-SpentFuel

This repository contains scripts used to implement a Gaussian Process Regression model for Spent Fuel nuclides.

# Kernel_Trainer.py:
This file allows for the computation of Kernels. Currently only support for the Anistropic Squared Exponential Kernel is provided.

# Model_Comparer.py:
With this file a comparison between GP models and Cubic Spline models can be made, provided the Kernels have been precomputed. 

# ---------------------------------------------------------------------------------------- #

# Kernels-Mass & Training_Sets:
These folders contain the Kernels, Training and Testing Sets used for the implementation of GP models of a CANDU 6 reactor. These models have been used for my paper on  ''Gaussian Processes for Surrogate Modeling of Discharged Fuel Nuclide Compositions''  in Annals of Nuclear Energy
