
import tempfile
import gym
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from imitation.algorithms import dagger
from imitation.policies import serialize
from imitation.util import util
from imitation.data import rollout


ENV_NAME = "CartPole-v1"
EXPERT_POLICY_PATH = "tests/data/expert_models/cartpole_0/policies/final/"
N_ROUNDS = 2
N_TRAJECTORIES_PER_ROUND = 5
N_ENVIRONMENTS = 1
SCRATCH_DIRECTORY = "experiments/scratch_dir/"
MIN_N_EPISODES = 15


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
    average_return_before_training = return_when_running_policy_on_env(agent_policy, vector_env)
    print(f'Average return before training: {average_return_before_training}')

    print('training DAgger...')
    train_dagger(vector_expert_policy, trainer, vector_env)

    print("running DAgger after training...")
    trained_policy = trainer.bc_trainer.policy
    average_return_after_training = return_when_running_policy_on_env(trained_policy, vector_env, render=False)

    print(f'Average return after training: {average_return_after_training}')


def return_when_running_policy_on_env(policy: BasePolicy, env: VecEnv, render=False):
    # # if we use it like in the tests (averaging over multiple trajectories/envs):
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
        observation, reward, done, _ = env.step(action)
        total_return += reward
        if done[0]:  # when the first env is done. (we don't care about the  others here)
            break
        step_n += 1

    # check that the environment is done as soon as the reward is not 1 anymore:
    assert step_n == int(total_return)
    env.close()

    return total_return


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


def make_trainer():
    beta_schedule = dagger.LinearBetaSchedule(1)
    env = gym.make(ENV_NAME)
    env.seed(42)
    tmpdir = tempfile.TemporaryDirectory()
    return dagger.DAggerTrainer(
        env,
        tmpdir.name,
        beta_schedule,
        optimizer_kwargs=dict(lr=1e-3),
    )


if __name__ == '__main__':
    run_and_show()
