from abc import ABC, abstractmethod
from gymnasium.spaces import Space


class Policy(ABC):
    def __init__(self, observation_space: Space, action_space: Space):
        self.observation_space: Space = observation_space
        self.action_space: Space = action_space

    @abstractmethod
    def action(self, observation):
        pass

    def reset(self):
        pass
