
import matplotlib.pyplot as plt
from torch import eye

def plot(trajectories, environment, size):  
    _, ax = plt.subplots(1, 2)
    traj = trajectories.sum(0).view(size, size)
    env = environment.reward(eye(environment.state_dim)).view(size, size)

    ax[0].matshow(traj.numpy())
    ax[0].set_title("Trajectories")
    ax[1].matshow(env.numpy())
    ax[1].set_title("Environment")
    plt.savefig("./figures/results.png")