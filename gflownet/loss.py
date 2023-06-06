import torch


def trajectory_balance_loss(total_flow, rewards, forward_probs, backward_probs):
    """
    Computes the mean trajectory balance loss for a collection of samples. For
    more information, see Bengio et al. (2022): https://arxiv.org/abs/2201.13259
    
    Args:
        total_flow: The estimated total flow used by the GFlowNet when drawing
        the collection of samples for which the loss should be computed. This
        is a normalizing constant of a Markovian flow F, where F(x) = R(x), and
        is an estimator of F(s0), where s0 is the initial state of a trajectory. 
        
        rewards: The rewards associated with the final state of each of the
        samples.
        
        forward_probs: The forward probabilities associated with the trajectory of
        each sample (i.e. the probabilities of the actions actually taken in
        each trajectory).
        
        backward_probs: The backward probabilities associated with each trajectory.
    """
    num = total_flow * torch.prod(forward_probs, dim=1)
    denom = rewards * torch.prod(backward_probs, dim=1)
    loss = torch.log(num / denom)**2
    return loss.mean()