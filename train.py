import torch
import torch.nn.functional as F
from tqdm import tqdm
from gflownet.loss import trajectory_balance_loss


def train_gflownet(environment, model, optimizer, batch_size, n_epochs):
    for i in (epoch := tqdm(range(n_epochs))):
        optimizer.zero_grad()

        # forward pass
        s0 = F.one_hot(torch.zeros(batch_size).long(), environment.state_dim).float()
        _, log = model.sample_states(s0, return_log=True)

        # compute loss and perform back-propagation
        loss = trajectory_balance_loss(log.total_flow,
                                       log.rewards,
                                       log.forward_probs,
                                       log.backward_probs)
        loss.backward()

        # update optimizer
        optimizer.step()
        
        # print training statistics
        if i % 10 == 0: epoch.set_description(f"Loss: {loss.item():.3f}")
    