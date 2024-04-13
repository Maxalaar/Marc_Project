from environments.identification_management.Identification_management import IdentificationManagement
import gymnasium as gym
import random

from environments.identification_management.configuration import default_environment_configuration
from policies.abstract import Policy
from policies.combination import Combination
from policies.random import Random
from policies.voracious import Voracious


def play_episode(environment: gym.Env, policy: Policy):
    observation, information = environment.reset()
    policy.reset()
    total_reward = 0

    continue_play = True
    while continue_play:
        action = policy.action(observation)
        observation, reward, terminated, truncated, information = environment.step(action)
        total_reward += reward

        if terminated is True or truncated is True:
            continue_play = False

    return total_reward


def play_iteration(environment: gym.Env, policy: Policy, number_episodes: int):
    total_reward: float = 0
    for _ in range(number_episodes):
        total_reward += play_episode(environment, policy)
    reward_episode_mean = total_reward / float(number_episodes)
    return reward_episode_mean


if __name__ == '__main__':
    environment_configuration: dict = default_environment_configuration
    environment_configuration['number_messages'] = 300
    environment_configuration['maximum_energy'] = 30
    # environment_configuration['render_mode'] = 'text'   # text or None

    identification_management: gym.Env = IdentificationManagement(environment_configuration=environment_configuration)
    random: Policy = Random(identification_management.observation_space, identification_management.action_space)
    voracious: Policy = Voracious(identification_management.observation_space, identification_management.action_space)
    maxime: Policy = Combination(identification_management.observation_space, identification_management.action_space)

    # print(f'Episode mean reward for random policy : {play_iteration(identification_management, random, 100)}')
    print(f'Episode mean reward for combination policy : {play_iteration(identification_management, maxime, 100)}')
    print(f'Episode mean reward for voracious policy : {play_iteration(identification_management, voracious, 100)}')

