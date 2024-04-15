import random
from typing import Callable, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import warnings
import json

from environments.identification_management.configuration import default_environment_configuration


class Message:
    def __init__(self, criticality: float, trust: float, is_real_source: bool):
        self.criticality: float = criticality
        self.trust: float = trust
        self.is_real_source: bool = is_real_source

    def get_information(self):
        return {'criticality': self.criticality, 'trust': self.trust, 'is_real_source': self.is_real_source}


class IdentificationManagement(gym.Env):
    def __init__(self, environment_configuration=None):
        if environment_configuration is None:
            environment_configuration = {}

        self.render_mode: Union[str, None] = environment_configuration.get(
            'render_mode', default_environment_configuration['render_mode']
        )

        self.current_action: Union[dict, None] = None
        self.messages: Union[list[Message], None] = None
        self.message_configuration: Callable = environment_configuration.get(
            'message_configuration', default_environment_configuration['message_configuration']
        )

        self.matrix_identification_factors: np.ndarray = environment_configuration.get(
            'matrix_identification_factors', default_environment_configuration['matrix_identification_factors'])
        self.number_identification_factors: int = self.matrix_identification_factors.shape[0]
        self.response_identification_factors: Union[np.ndarray, None] = None
        self.number_messages: int = environment_configuration.get(
            'number_messages', default_environment_configuration['number_messages'])
        self.position_current_message: Union[int, None] = None
        self.current_message: Union[Message, None] = None
        self.maximum_energy: float = environment_configuration.get(
            'maximum_energy', default_environment_configuration['maximum_energy'])
        self.current_energy: Union[float, None] = None

        self.is_terminated: Union[bool, None] = None
        self.is_truncated: Union[bool, None] = None

        self.observation_space = spaces.Dict(
            {
                'matrix_identification_factors': spaces.Box(
                    low=np.finfo(np.float64).min/2,
                    high=np.finfo(np.float64).max/2,
                    shape=self.matrix_identification_factors.shape,
                    dtype=np.float64),
                'response_identification_factors': spaces.Box(
                    low=-1,
                    high=1,
                    shape=(self.number_identification_factors,),
                    dtype=np.int32),
                'current_energy': spaces.Box(
                    low=0,
                    high=np.finfo(np.float64).max/2,
                    shape=(1,),
                    dtype=np.float64),
                'position_current_message': spaces.Box(
                    low=0,
                    high=np.iinfo(np.int32).max/2,
                    shape=(1,),
                    dtype=np.int32),
                'number_messages': spaces.Box(
                    low=0,
                    high=np.iinfo(np.int32).max/2,
                    shape=(1,),
                    dtype=np.int32),
                'current_message_criticality': spaces.Box(
                    low=np.finfo(np.float64).min/2,
                    high=np.finfo(np.float64).max/2,
                    shape=(1,),
                    dtype=np.float64),
                'current_message_trust': spaces.Box(
                    low=np.finfo(np.float64).min/2,
                    high=np.finfo(np.float64).max/2,
                    shape=(1,),
                    dtype=np.float64),
            }
        )
        self.action_space = spaces.Dict(
            {
                'is_real_source': spaces.Discrete(
                    n=3),
                'calling_identification_factors': spaces.MultiDiscrete(
                    np.full((self.number_identification_factors,), 2)),
            }
        )

        self.int_to_bool = {-1: False, 1: True}
        self.bool_to_int = {False: np.array([-1]), True: np.array(1)}

    def reset(self, seed=None, options=None):
        self.current_action = None
        self.messages = []

        for _ in range(self.number_messages):
            self._create_message()

        self.current_energy = self.maximum_energy
        self.position_current_message = 0
        self.current_message = self.messages[self.position_current_message]
        self.response_identification_factors = np.zeros(self.number_identification_factors, dtype=np.int32)

        self.is_terminated = False
        self.is_truncated = False

        return self._get_observation(), self._get_information()

    def _create_message(self):
        self.messages.append(Message(**self.message_configuration()))

    def step(self, action: dict):
        self.current_action = action
        action_is_real_source: int = int(action['is_real_source']) - 1
        action_calling_identification_factors: np.ndarray = action['calling_identification_factors']
        reward: Union[float, None] = None

        if action_is_real_source != 0:
            if self.int_to_bool[action_is_real_source] == self.current_message.is_real_source:
                reward = self.current_message.criticality * self.current_message.trust
            elif self.int_to_bool[action_is_real_source] != self.current_message.is_real_source:
                reward = -1 * self.current_message.criticality * self.current_message.trust

            if self.position_current_message >= self.number_messages - 1:
                self.is_terminated = True
            else:
                self._next_message()

        else:
            made_mistake = self._calling_identification_factors(action_calling_identification_factors)
            reward = 0
            if made_mistake:
                reward = -1 * self.current_message.criticality * self.current_message.trust
                if self.position_current_message >= self.number_messages - 1:
                    self.is_terminated = True
                else:
                    self._next_message()

        if self.render_mode == 'text':
            self.render()

        return self._get_observation(), reward, self.is_terminated, self.is_truncated, self._get_information()

    def _get_information(self):
        information = {'observation': self._get_observation()}
        information.update({'message': self.current_message.get_information()})
        if self.current_action is not None:
            information.update({'action': self.current_action})

        return information

    def _get_observation(self):
        observation = {
            'matrix_identification_factors': self.matrix_identification_factors,
            'response_identification_factors': self.response_identification_factors,
            'current_energy': np.array([self.current_energy], dtype=np.float64),
            'number_messages': np.array([self.number_messages]),
            'position_current_message': np.array([self.position_current_message]),
            'current_message_criticality': np.array([self.current_message.criticality]),
            'current_message_trust': np.array([self.current_message.trust]),
        }
        return observation

    def _next_message(self):
        self.position_current_message += 1
        self.response_identification_factors = np.zeros(self.number_identification_factors, dtype=np.int32)
        self.current_message = self.messages[self.position_current_message]

    def _calling_identification_factors(self, calling: np.ndarray) -> bool:
        made_mistake: bool = False
        for i in range(self.number_identification_factors):
            if calling[i] == 1 and self.response_identification_factors[i] == 0:
                factor_energy_cost = self.matrix_identification_factors[i][0]
                factor_percentage_correct_responses = self.matrix_identification_factors[i][1]

                if self.current_energy >= factor_energy_cost:
                    self.current_energy -= factor_energy_cost
                    if np.random.uniform(0, 1) < factor_percentage_correct_responses:
                        self.response_identification_factors[i] = self.bool_to_int[self.current_message.is_real_source]
                    else:
                        self.response_identification_factors[i] = self.bool_to_int[not self.current_message.is_real_source]
                else:
                    made_mistake = True
                    print(f'The agent is attempting to call identification factor {i} for which it does not '
                                  f'have enough energy.')

            elif calling[i] == 1 and self.response_identification_factors[i] != 0:
                made_mistake = True
                print(f'The agent is attempting to call the identification factor {i} that has already been '
                              f'called for this message.')
        return made_mistake

    def render(self):
        print()
        print('-- Render --')
        def numpy_array_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convertir le tableau numpy en une liste standard
            raise TypeError("Type not serializable")

        print(json.dumps(self._get_information(), indent=4, default=numpy_array_serializer))
