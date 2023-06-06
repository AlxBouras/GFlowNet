import torch
import torch.nn.functional as F
from torch.optim import Adam
from train import train_gflownet
from gflownet.gflownet import GFlowNet
from policy import ForwardPolicy, BackwardPolicy
from grid import Grid
from utils import plot


if __name__ == "__main__":
    # set random seed
    torch.manual_seed(42)

    # initialize environement
    size = 16
    env = Grid(size=size)

    # hyperparameters
    n_epochs = 1000
    batch_size = 256
    hidden_dim = 128
    learning_rate = 5e-3

    # initialize model
    forward_policy = ForwardPolicy(env.state_dim, hidden_dim, n_actions=env.n_actions)
    backward_policy = BackwardPolicy(env.state_dim, n_actions=env.n_actions)
    model = GFlowNet(forward_policy, backward_policy, env)
    optimizer = Adam(model.parameters(), learning_rate)

    # train model (online setting)
    train_gflownet(env, model, optimizer, batch_size, n_epochs)

    # sample complete trajectories with trained model
    s0 = F.one_hot(torch.zeros(10**4).long(), env.state_dim).float()
    states = model.sample_states(s0, return_log=False)
    plot(states, env, size)