from ray.tune.registry import register_env

from environments.identification_management.Identification_management import IdentificationManagement


def register_environments():
    register_env(name='IdentificationManagement', env_creator=IdentificationManagement)
