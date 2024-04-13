from typing import Union

from gymnasium.spaces import Space
import numpy as np
from policies.abstract import Policy


class Voracious(Policy):
    def __init__(self, observation_space: Space, action_space: Space):
        super().__init__(observation_space, action_space)
        self.is_first_time_seeing_message: Union[bool, None] = None

    def reset(self):
        self.is_first_time_seeing_message: Union[bool, None] = True

    def action(self, observation):
        action = {}

        # First column: energy cost, Second column: percentage of correct responses
        matrix_identification_factors = observation['matrix_identification_factors']

        # Response of the authentication factor call: 0 - not called by the agent, 1 - factor indicates authentication is good, -1 - factor indicates authentication is bad
        response_identification_factors = observation['response_identification_factors']

        number_factor = matrix_identification_factors.shape[0]
        indices_order_call_factors = np.argsort(matrix_identification_factors[:, 1])[::-1]

        if self.is_first_time_seeing_message:
            # Calculation of the budget for this message
            current_energy = observation['current_energy']
            number_messages_remaining_before_end = observation['number_messages'] - observation['position_current_message']
            estimation_average_message_criticality = 0.5
            estimation_average_message_confidence = 0.5
            energy_budget = (observation['current_message_criticality'] * observation['current_message_trust']) / (number_messages_remaining_before_end * estimation_average_message_criticality * estimation_average_message_confidence) * current_energy
            if energy_budget > current_energy:
                energy_budget = current_energy

            # We are looking for the maximum criteria we can call with this budget, starting with the most profitable ones
            identification_factors_call = []

            calling_one_factor = False
            for index in indices_order_call_factors:
                factor_energy_cost = matrix_identification_factors[index][0]
                if energy_budget >= factor_energy_cost and calling_one_factor is False:
                    identification_factors_call.append(index)
                    energy_budget -= factor_energy_cost
                    calling_one_factor = True

            calling_identification_factors = np.zeros(number_factor, dtype=np.int32)
            calling_identification_factors[identification_factors_call] = 1
            action = {
                'is_real_source': 0,
                'calling_identification_factors': calling_identification_factors,
            }

            self.is_first_time_seeing_message = False
        else:
            calling_identification_factors = np.zeros(number_factor, dtype=np.int32)

            if np.any(response_identification_factors == 1):
                is_real_source = 1
            else:
                is_real_source = -1

            action = {
                'is_real_source': is_real_source,
                'calling_identification_factors': calling_identification_factors,
            }

            self.is_first_time_seeing_message = True

        return action
