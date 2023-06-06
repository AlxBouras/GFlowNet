import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, ones


class ForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
    
    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
class BackwardPolicy:
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.size = int(state_dim**0.5)
    
    def __call__(self, s):
        idx = s.argmax(-1)
        at_top_edge = idx < self.size
        at_left_edge = (idx > 0) & (idx % self.size == 0)
        
        probs = 0.5 * ones(len(s), self.n_actions)
        probs[at_left_edge] = Tensor([1, 0, 0])
        probs[at_top_edge] = Tensor([0, 1, 0])
        probs[:, -1] = 0 # disregard termination
        return probs