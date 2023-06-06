import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical
from gflownet.log import Log


class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env):
        """
        Initializes a GFlowNet using the specified forward and backward policies
        acting over an environment, i.e. a state space and a reward function.
        
        Args:
            forward_policy: A policy network taking as input a state and
            outputting a vector of probabilities over actions.
            
            backward_policy: A policy network (or fixed function) taking as
            input a state and outputting a vector of probabilities over the
            actions which led to that state.
            
            env: An environment defining a state space and an associated reward
            function.
        """
        super().__init__()
        self.total_flow = Parameter(torch.ones(1))
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.env = env
    
    def mask_and_normalize(self, states, action_probabilities):
        """
        Masks a vector of action probabilities to avoid illegal actions
        (i.e. actions that lead outside the state space).
        
        Args:
            states: An mXd matrix representing m states.
            
            action_probabilities: An mXa matrix of action probabilities.
        """
        action_probabilities = self.env.mask(states) * action_probabilities
        return action_probabilities / action_probabilities.sum(1).unsqueeze(1)
    
    def forward_probabilities(self, states):
        """
        Returns a vector of probabilities over actions in a given state.
        
        Args:
            states: An mXd matrix representing m states.
        """
        probabilities = self.forward_policy(states)
        return self.mask_and_normalize(states, probabilities)
    
    def sample_states(self, init_states, return_log=False):
        """
        Samples and returns a collection of final states from the GFlowNet.
        
        Args:
            init_states: An mXd matrix of initial states.
            
            return_log: Return an object containing information about the
            sampling process (e.g. the trajectory of each sample, the forward
            and backward probabilities, the actions taken, etc.).
        """
        states = init_states.clone()
        done = torch.BoolTensor([False] * len(states))
        log = Log(init_states, self.backward_policy, self.total_flow, self.env) if return_log else None

        while not done.all():
            probs = self.forward_probabilities(states[~done])
            actions = Categorical(probs).sample()
            states[~done] = self.env.update(states[~done], actions)
            
            if return_log:
                log.log(states, probs, actions, done)
                
            terminated = actions == probs.shape[-1] - 1
            done[~done] = terminated
        
        return (states,log) if return_log else states
    
    def evaluate_trajectories(self, sample_trajectories, actions):
        """
        Returns the GFlowNet's estimated forward probabilities, backward
        probabilities, and rewards for a collection of trajectories. This is
        useful in an offline learning context where samples are drawn according
        to another policy (e.g. random policy) and are used to train the model.
        
        Args:
            sample_trajectories: The trajectory of each sample.
            
            actions: The actions that produced the trajectories in sample_trajectories.
        """
        n_samples = len(sample_trajectories)
        sample_trajectories = sample_trajectories.reshape(-1, sample_trajectories.shape[-1])
        actions = actions.flatten()
        final_states = sample_trajectories[actions == self.env.num_actions - 1]
        zero_to_n = torch.arange(len(actions))

        # Get the corresponding forward probabilities for the sampled trajectories
        forward_probs = self.forward_probabilities(sample_trajectories)
        forward_probs = torch.where(actions == -1, 1, forward_probs[zero_to_n, actions])
        forward_probs = forward_probs.reshape(n_samples, -1)

        actions = actions.reshape(n_samples, -1)[:, :-1].flatten()

        # Get the corresponding backward probabilities for the sampled trajectories
        backward_probs = self.backward_policy(sample_trajectories)
        backward_probs = backward_probs.reshape(n_samples, -1, backward_probs.shape[-1])
        backward_probs = backward_probs[:, -1, :].reshape(-1, backward_probs.shape[2])
        backward_probs = torch.where((actions == -1) or (actions == 2), 1,
                                     backward_probs[zero_to_n[:-n_samples], actions])
        
        rewards = self.env.reward(final_states)

        return forward_probs, backward_probs, rewards