
# import glob
# import os
import tempfile
import typing

import gym
# import numpy as np
import pytest
import torch
# from stable_baselines3.common import policies
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from imitation.algorithms import dagger  #, bc
from imitation.algorithms.dagger import DAggerTrainer
# from imitation.data import rollout
from imitation.policies import serialize
from imitation.util import util

ENV_NAME = "CartPole-v1"
EXPERT_POLICY_PATH = "tests/data/expert_models/cartpole_0/policies/final/"
N_ROUNDS = 2
N_TRAJECTORIES_PER_ROUND = 5
# N_ENVIRONMENTS = 1
SCRATCH_DIRECTORY = "experiments/scratch_dir/"


def run_and_show():

    print('Making DAggerTrainer and policy...')
    trainer = make_trainer(SCRATCH_DIRECTORY)
    agent_policy = trainer.bc_trainer.policy

    print('Making environment...')
    vector_env: DummyVecEnv = util.make_vec_env(ENV_NAME)  # type:ignore
    dummy_vec_env: DummyVecEnv = vector_env.unwrapped
    single_env = dummy_vec_env.envs[0].unwrapped

    print('loading expert policy...')
    vector_expert_policy = serialize.load_policy("ppo", EXPERT_POLICY_PATH, vector_env)

    print('running DAgger before training...')
    average_reward_before_training = evaluate_dagger_policy(agent_policy, vector_env)
    print(f'Average reward before training: {average_reward_before_training}')

    print('training DAgger...')
    train_dagger(vector_expert_policy, trainer, single_env)

    print("running DAgger after training...")
    trained_policy = trainer.bc_trainer.policy
    average_reward_after_training = evaluate_dagger_policy(trained_policy, vector_env, render=False)

    print(f'Average reward after training: {average_reward_after_training}')


def evaluate_dagger_policy(policy: BasePolicy, env: VecEnv, render=False):
    observations = env.reset()
    average_reward = 0
    n = 1  # counter to compute the average reward
    while True:
        if render:
            env.render()
        action, _ = policy.predict(observations)
        observation, reward, done, _ = env.step(action)
        average_reward = (average_reward*(n-1) + reward)/n
        if any(done):  # when some env is done.
            break
        n += 1
    env.close()
    return average_reward


def train_dagger(expert_policy, trainer, env, render=False):
    for i in range(N_ROUNDS):
        # roll out a few trajectories for dataset, then train for a few steps
        collector = trainer.get_trajectory_collector()  # for fetching demonstrations
        for _ in range(N_TRAJECTORIES_PER_ROUND):
            obs = collector.reset()  # first observation of a new trajectory.
            done = False
            while not done:
                if render:
                    env.render()
                (expert_action,), _ = expert_policy.predict(
                    obs[None], deterministic=True
                )  # collects expert actions/predictions
                obs, _, done, _ = collector.step(expert_action)  # using the expert_action in most cases
                # while randomly injecting the actual policy.
        trainer.extend_and_update(n_epochs=1)

def make_trainer(tmpdir, beta_schedule=dagger.LinearBetaSchedule(1)):
    env = gym.make(ENV_NAME)
    env.seed(42)
    tmpdir = tempfile.TemporaryDirectory()
    return dagger.DAggerTrainer(
        env,
        tmpdir.name,
        beta_schedule,
        optimizer_kwargs=dict(lr=1e-3),
    )


@pytest.fixture(params=[None, dagger.LinearBetaSchedule(1)])
def trainer(request, tmpdir):
    beta_schedule = request.param
    return make_trainer(tmpdir, beta_schedule)



if __name__ == '__main__':
    run_and_show()
