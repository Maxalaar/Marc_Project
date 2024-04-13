import random
import numpy as np


default_message_configuration = lambda: {
    'criticality': random.uniform(0, 1),
    'trust': random.uniform(0, 1),
    'is_real_source': random.choice([True, False]),
}

default_matrix_identification_factors = np.array([
    [0.8, 0.8],
    [0.4, 0.7],
    [0.6, 0.75],
    [0.2, 0.6],
])

def random_matrix_identification_factors(number_factors):
    energy_costs = np.random.uniform(0.5, 1, size=(number_factors, 1))
    probability_correct_prediction = np.random.uniform(0.5, 1, size=(number_factors, 1))
    matrix = np.hstack((energy_costs, probability_correct_prediction))
    return matrix


default_environment_configuration: dict = {
    'message_configuration': default_message_configuration,
    'matrix_identification_factors': default_matrix_identification_factors,
    'number_messages': 1000,
    'maximum_energy': 500,
    'render_mode': False,
}


def random_environment_configuration():
    environment_configuration: dict = {
        'message_configuration': default_message_configuration,
        'matrix_identification_factors': random_matrix_identification_factors(random.randint(2, 10)),
        'number_messages': random.randint(500, 2000),
        'maximum_energy': random.uniform(150, 450),
        'render_mode': False,
    }
    return environment_configuration
