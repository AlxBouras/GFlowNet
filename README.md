# GFlowNet implementation

This repo contains an implementation of a GFlowNet using PyTorch. GFlowNets were first proposed by Bengio et al. in the paper ["Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation"](https://arxiv.org/abs/2106.04399) (2021).

The model is trained using online learning (i.e. by evaluating samples drawn from the model's behavior policy rather than a fixed set of samples drawn from another policy) and the [trajectory balance loss](https://arxiv.org/abs/2201.13259). We evaluate the model's performance using the grid domain of the original paper.
