
import tempfile
import gym
import numpy as np
from imitation.algorithms.dagger import _save_trajectory
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from imitation.algorithms import dagger
from imitation.policies import serialize
from imitation.util import util
import os
from imitation.data import rollout

from dagger_experiment import N_TRAJECTORIES_PER_ROUND, train_dagger, make_trainer



ENV_NAME = "CartPole-v1"
EXPERT_POLICY_PATH = "tests/data/expert_models/cartpole_0/policies/final/"

TOTAL_STEPS = 30000
N_POINTS_TO_PLOT = 10
MAX_STEPS_PER_TRAJECTORY = 505
N_TRAJECTORIES_PER_ROUND = 1



TOTAL_STEPS = 10000
MAX_DECAY_STEPS = 5000
N_POINTS_TO_PLOT = 10
MAX_STEPS_PER_TRAJECTORY = 200

def run_rampdown_experiment(filename='test2_gets_knocked', gets_knocked=False):

    episode_lengths = []
    times_failed = []
    failed = []
    vector_env: DummyVecEnv = util.make_vec_env(ENV_NAME, n_envs=1)  # type:ignore

    # vector_expert_policy = serialize.load_policy("ppo", EXPERT_POLICY_PATH, vector_env)

    expert_policy = cartpole_safe_mentor
    for point in range(N_POINTS_TO_PLOT):

        decay_end_point = MAX_DECAY_STEPS // (point + 1)

        rampdown_rounds = decay_end_point // (MAX_STEPS_PER_TRAJECTORY * N_TRAJECTORIES_PER_ROUND)
        n_rounds = TOTAL_STEPS // (MAX_STEPS_PER_TRAJECTORY * N_TRAJECTORIES_PER_ROUND)


        trainer = make_trainer(rampdown_rounds=rampdown_rounds)
        
        ep_lens, fails, failed_trajectories = train_dagger(expert_policy, trainer, vector_env,
                                        N_ROUNDS=n_rounds, 
                                        N_TRAJECTORIES_PER_ROUND=N_TRAJECTORIES_PER_ROUND,
                                        MAX_STEPS_PER_TRAJECTORY=MAX_STEPS_PER_TRAJECTORY,
                                        gets_knocked=gets_knocked)

        episode_lengths.append(ep_lens)
        times_failed.append(fails)
        failed.append(failed_trajectories)

    print(times_failed)
    print(episode_lengths)
    print(failed)
    np.savez(filename, times_failed=times_failed, 
            episode_lengths=episode_lengths,
            failed=failed,
            MAX_STEPS_PER_TRAJECTORY=MAX_STEPS_PER_TRAJECTORY,
            N_TRAJECTORIES_PER_ROUND=N_TRAJECTORIES_PER_ROUND,
            TOTAL_STEPS=TOTAL_STEPS,
            N_POINTS_TO_PLOT=N_POINTS_TO_PLOT)

    return episode_lengths, times_failed, failed



def cartpole_safe_mentor(state, kwargs=None):
    """A safe mentor for cartpole with un-normalised inputs

    Stands at the centre

    Originated from Michael Cohen.
    """
    x = state[0] / 9.6
    v = state[1] / 20
    theta = state[2] / 0.836
    v_target = max(min(-x * 0.5, 0.01), -0.01)
    theta_target = max(min(- (v - v_target) * 4, 0.2), -0.2)
    w = state[3] / 2
    w_target = max(min(- (theta - theta_target) * 0.9, 0.1), -0.1)
    return 0 if w < w_target else 1

if __name__ == '__main__':
    run_rampdown_experiment(filename='rampdown_exp1', gets_knocked=False)
    run_rampdown_experiment(filename='rampdown_knocked_exp1', gets_knocked=True)


