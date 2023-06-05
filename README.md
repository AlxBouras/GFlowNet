# GFlowNet implementation

This repo contains an implementation of a GFlowNet using PyTorch. GFlowNets were first proposed by Bengio et al. in the paper [Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation](https://arxiv.org/abs/2106.04399) (2021).

The model is trained using online learning (i.e. by evaluating samples drawn from the model's behavior policy rather than a fixed set of samples drawn from another policy) and the [trajectory balance loss](https://arxiv.org/abs/2201.13259). We evaluate the model's performance using the grid domain of the original paper.

The model training follows the following steps:

1. Initialize the grid environment using a grid size.
2. Define a policy network taking a state vector as input and returning a vector of probabilities over possible actions. (In the grid domain, there are three possible actions: **Down**, **Right**, and **Terminate**.)
3. Define a backward policy. In this case, the policy is not estimated but fixed to 0.5 for all parent states (except when there is only one parent state).

Run **main.py** to start the GFlowNet training and use the model once trained.
