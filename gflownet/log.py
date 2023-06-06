import torch


class Log:
    def __init__(self, init_states, backward_policy, total_flow, env):
        """
        Initializes a stats object to record sampling statistics from a
        GFlowNet (e.g. trajectories, forward and backward probabilities,
        actions, etc.)
        
        Args:
            init_states: The initial state of a collection of samples.
            
            backward_policy: The backward policy used to estimate the backward
            probabilities associated with each sample's trajectory.
            
            total_flow: The estimated total flow used by the GFlowNet during
            sampling.
            
            env: The environment (i.e. state space and reward function) from
            which samples are drawn.
        """
        self.backward_policy = backward_policy
        self.total_flow = total_flow
        self.env = env
        self._trajectories = [init_states.view(len(init_states), 1, -1)]
        self._forward_probs = []
        self._backward_probs = None
        self._actions = []
        self.rewards = torch.zeros(len(init_states))
        self.n_samples = init_states.shape[0]

    def log(self, states, forward_probs, actions, done):
        """
        Logs relevant information about each sampling step
        
        Args:
            states: An mxd matrix containing the current states of complete and
            incomplete samples/trajectories.
            
            forward_probs: An nxd matrix containing the forward probabilities output by the
            GFlowNet for the given states.
            
            actions: An nx1 vector containing the actions taken by the GFlowNet
            in the given states.
            
            done: An nx1 Bool vector indicating which samples are complete
            (True) and which are incomplete (False).
        """
        complete_trajectory = actions == forward_probs.shape[-1] - 1
        active, finished = ~done, ~done
        active[active == True] = ~complete_trajectory
        finished[finished == True] = complete_trajectory

        # log trajectory states
        states_log = self._trajectories[-1].squeeze(1).clone()
        states_log[active] = states[active]
        self._trajectories.append(states_log.view(self.n_samples, 1, -1))

        # log forward probabilities
        forward_probs_log = torch.ones(self.n_samples, 1)
        forward_probs_log[~done] = forward_probs.gather(1, actions.unsqueeze(1))
        self._forward_probs.append(forward_probs_log)

        # log taken actions
        actions_log = -torch.ones(self.n_samples, 1).long()
        actions_log[~done] = actions.unsqueeze(1)
        self._actions.append(actions_log)

        # log corresponding rewards for trajectories which have just been completed
        self.rewards[finished] = self.env.reward(states[finished])

    @property
    def trajectories(self):
        if type(self._trajectories) is list:
            self._trajectories = torch.cat(self._trajectories, dim=1)[:, :-1, :]
        return self._trajectories
    
    @property
    def forward_probs(self):
        if type(self._forward_probs) is list:
            self._forward_probs = torch.cat(self._forward_probs, dim=1)
        return self._forward_probs
    
    @property
    def actions(self):
        if type(self._actions) is list:
            self._actions = torch.cat(self._actions, dim=1)
        return self._actions
    
    @property
    def backward_probs(self):
        if self._backward_probs is not None:
            return self._backward_probs
        
        s = self.trajectories[:, 1:, :].reshape(-1, self.env.state_dim)
        prev_s = self.trajectories[:, :-1, :].reshape(-1, self.env.state_dim)
        actions = self.actions[:, :-1].flatten()
        
        terminated = (actions == -1) | (actions == self.env.n_actions - 1)
        zero_to_n = torch.arange(len(terminated))
        backward_probs = self.backward_policy(s) * self.env.mask(prev_s)
        backward_probs = torch.where(terminated, torch.tensor(1.), backward_probs[zero_to_n, actions])
        self._backward_probs = backward_probs.reshape(self.n_samples, -1)
        
        return self._backward_probs