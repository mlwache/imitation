
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


ENV_NAME = "CartPole-v1"
EXPERT_POLICY_PATH = "tests/data/expert_models/cartpole_0/policies/final/"
N_ROUNDS = 10
N_TRAJECTORIES_PER_ROUND = 1
N_ENVIRONMENTS = 1
SCRATCH_DIRECTORY = "experiments/scratch_dir/"
MIN_N_EPISODES = 15  # only used by rollouts.py
RAMPDOWN_ROUNDS: int = 100#_ROUNDS - 1  # This way the agent acts alone in the end.


def run_and_show():

    print('Making DAggerTrainer and policy...')
    trainer = make_trainer()
    agent_policy = trainer.bc_trainer.policy

    print('Making environment...')
    vector_env: DummyVecEnv = util.make_vec_env(ENV_NAME, n_envs=N_ENVIRONMENTS)  # type:ignore
    # dummy_vec_env: DummyVecEnv = vector_env.unwrapped
    # single_env = dummy_vec_env.envs[0].unwrapped

    print('loading expert policy...')
    vector_expert_policy = serialize.load_policy("ppo", EXPERT_POLICY_PATH, vector_env)

    print('running DAgger before training...')
    return_before_training = return_when_running_policy_on_env(agent_policy, vector_env)
    print(f'Return before training: {return_before_training}')

    print('training DAgger...')
    train_dagger(vector_expert_policy, trainer, vector_env)

    print("running DAgger after training...")
    trained_policy = trainer.bc_trainer.policy
    return_after_training = return_when_running_policy_on_env(trained_policy, vector_env, render=False)

    print(f'Return after training: {return_after_training}')


def return_when_running_policy_on_env(policy: BasePolicy, env: VecEnv, render=False):
    # if we use it like in the tests (averaging over multiple trajectories/envs):
    # mean_return = rollout.mean_return(
    #     policy,
    #     env,
    #     sample_until=rollout.min_episodes(MIN_N_EPISODES),
    # )
    # return mean_return
    observations = env.reset()
    total_return = 0
    step_n = 1
    while True:
        if render:
            env.render()
        action, _ = policy.predict(observations)
        observations, reward, done, _ = env.step(action)
        total_return += reward
        if done[0]:  # when the first env is done. (we don't care about the  others here)
            break
        step_n += 1

    # check that the environment is done as soon as the reward is not 1 anymore:
    assert step_n == int(total_return)
    env.close()

    return total_return


def train_dagger(expert_policy, trainer, env, render=False, N_ROUNDS=N_ROUNDS,
                N_TRAJECTORIES_PER_ROUND=N_TRAJECTORIES_PER_ROUND,
                MAX_STEPS_PER_TRAJECTORY=505, gets_knocked=False
                ):
    episode_lengths = []
    failed = []
    times_failed = 0
    final_obs_from_last_trajectory = None
    for current_round in range(N_ROUNDS):
        episode_lengths.append([])
        print(f'starting round {current_round} out of {N_ROUNDS}!')
        # roll out a few trajectories for dataset, then train for a few steps
        collector = trainer.get_trajectory_collector()  # for fetching demonstrations
        for current_trajectory_within_round in range(N_TRAJECTORIES_PER_ROUND):

            # reset only for the first trajectory.
            if current_round == 0 and current_trajectory_within_round == 0:
                obs = collector.reset()
            else:
                obs = collector.reset(final_obs_from_last_trajectory)

            j = 0
            done = False
            while not done:
                if render:
                    env.render()
                # (expert_action,), _ = expert_policy.predict(
                #     obs[None], deterministic=True
                # )  # collects expert actions/predictions
                # print(obs)
                expert_action = expert_policy(obs)
                obs, _, done, _ = collector.step(expert_action, gets_knocked=gets_knocked)  # using the expert_action in most cases
                j += 1
                # while randomly injecting the actual policy.
                if done:  # done because it failed
                    assert collector.env._max_episode_steps == np.inf
                    times_failed += 1
                    failed.append(True) # Could change this to have a different list for each round
                    collector.reset()
                if j >= MAX_STEPS_PER_TRAJECTORY:
                    done = True
                    failed.append(False)
                    trajectory = collector.traj_accum.finish_trajectory()
                    timestamp = util.make_unique_timestamp()
                    trajectory_path = os.path.join(
                        collector.save_dir, "dagger-demo-" + timestamp + ".npz"
                    )
                    _save_trajectory(trajectory_path, trajectory)
            episode_lengths[current_round].append(j)
            final_obs_from_last_trajectory = obs
        trainer.extend_and_update(n_epochs=1)
    print("episode lengths: ", episode_lengths)
    print("times failed:", times_failed)

    return episode_lengths, times_failed, failed


def make_trainer(rampdown_rounds=RAMPDOWN_ROUNDS):
    beta_schedule = dagger.LinearBetaSchedule(rampdown_rounds)
    env = gym.make(ENV_NAME)
    env.seed(42)

    env._max_episode_steps = np.inf
    tmpdir = tempfile.TemporaryDirectory()
    return dagger.DAggerTrainer(
        env,
        tmpdir.name,
        beta_schedule,
        optimizer_kwargs=dict(lr=1e-3),
    )


if __name__ == '__main__':
    run_and_show()
