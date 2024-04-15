import warnings

import numpy as np

from environments.identification_management.Identification_management import IdentificationManagement
import gymnasium as gym
import random

from environments.identification_management.configuration import default_environment_configuration
from policies.abstract import Policy
from policies.brutal import Brutal
from policies.combination import Combination
from policies.random import Random
from policies.voracious import Voracious
from train_deep_policy import train_deep_policy


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
    # warnings.filterwarnings("ignore")
    environment_configuration: dict = default_environment_configuration
    environment_configuration['number_messages'] = 300
    environment_configuration['maximum_energy'] = 30

    # First column: energy cost, Second column: percentage of correct responses
    environment_configuration['matrix_identification_factors'] = np.array([
        [0.8, 0.8],
        [0.4, 0.7],
        [0.6, 0.75],
        [0.2, 0.6],
    ])

    # environment_configuration['render_mode'] = 'text'   # text or None

    identification_management: gym.Env = IdentificationManagement(environment_configuration=environment_configuration)
    # random: Policy = Random(identification_management.observation_space, identification_management.action_space)
    # brutal: Policy = Brutal(identification_management.observation_space, identification_management.action_space)
    # combination: Policy = Combination(identification_management.observation_space, identification_management.action_space)
    voracious: Policy = Voracious(identification_management.observation_space, identification_management.action_space)

    # print(f'Episode mean reward for random policy : {play_iteration(identification_management, random, 1)}')
    # print(f'Episode mean reward for brutal policy : {play_iteration(identification_management, brutal, 500)}')
    # print(f'Episode mean reward for combination policy : {play_iteration(identification_management, combination, 500)}')
    # print(f'Episode mean reward for voracious policy : {play_iteration(identification_management, voracious, 500)}')

    train_deep_policy(environment_name='IdentificationManagement', environment_configuration=environment_configuration)



