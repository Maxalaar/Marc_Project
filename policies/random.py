from gymnasium.spaces import Space

from policies.abstract import Policy


class Random(Policy):
    def __init__(self, observation_space: Space, action_space: Space):
        super().__init__(observation_space, action_space)

    def action(self, observation):
        return self.action_space.sample()
