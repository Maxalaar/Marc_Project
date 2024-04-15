import gymnasium
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from environments.register_environments import register_environments
from ray import air, tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig, PPO


def train_deep_policy(environment_name: str, environment_configuration: dict):

    # register_environments()
    # config = (  # 1. Configure the algorithm,
    #     PPOConfig()
    #     .environment(env=environment_name, env_config=environment_configuration)
    #     .rollouts(num_rollout_workers=2)
    #     .framework("torch")
    #     .training(model={"fcnet_hiddens": [248, 248, 248, 248]})
    #     .evaluation(evaluation_num_workers=1)
    # )
    #
    # algo = config.build()  # 2. build the algorithm,
    #
    # for _ in range(1_000_000):
    #     print(algo.train()['sampler_results']['episode_reward_mean'])  # 3. train it,
    #
    # algo.evaluate()

    ray.init(local_mode=False)
    register_environments()

    algorithm_configuration: AlgorithmConfig = (
        PPOConfig()
        .environment(env=environment_name, env_config=environment_configuration)
        .framework('torch')
        .training(
            model={"fcnet_hiddens": [248, 248, 248, 248]},
            train_batch_size=8000,
        )
        .rollouts(
            num_rollout_workers=2,
            batch_mode='complete_episodes'
        )
        .resources(
            num_gpus=1,
            num_learner_workers=3,
            num_cpus_per_worker=1,
            num_gpus_per_worker=0,
            # num_cpus_per_learner_worker=2,
            num_gpus_per_learner_worker=0.3,
        )
        .evaluation(evaluation_num_workers=1)
    )

    tuner = tune.Tuner(
        trainable=PPO,
        param_space=algorithm_configuration,
        run_config=air.RunConfig(
            name='exp_debug',
            storage_path='/home/maxalaar/PycharmProjects/Marc_Project/results',
            stop={
                'time_total_s': 60 * 60 * 5,
                # 'episode_reward_mean': 0.95,
            },
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_score_order='max',
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            )
        ),
    )

    tuner.fit()

    ray.shutdown()
