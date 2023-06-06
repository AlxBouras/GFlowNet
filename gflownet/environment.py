from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Abstract base class defining the signatures of the required functions to be
    implemented in a GFlowNet environment.
    """

    @abstractmethod
    def update(self, states, actions):
        """
        Takes as input state-action pairs and returns the resulting states.
        
        Args:
            states: An mxd matrix of state vectors
            
            actions: An mx1 vector of actions
        """
        pass

    @abstractmethod
    def mask(self, states):
        """
        Defines a mask to disallow certain actions given certain states.
        
        Args:
            states: An nxd matrix of state vectors
        """
        pass

    @abstractmethod
    def reward(self, states):
        """
        Defines a reward function, mapping states to rewards.
        
        Args:
            states: An mxd matrix of state vectors
        """
        pass