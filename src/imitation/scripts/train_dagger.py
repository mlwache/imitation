import logging
import os.path as osp
import pathlib
import pickle
from typing import Iterable, Mapping, Optional, Sequence, Union

import gym
import sacred
import torch as th
from gym import spaces
from sacred.observers import FileStorageObserver
from stable_baselines3.common import policies, vec_env
from torch.utils import data as th_data

from imitation.algorithms import dagger
from imitation.data import rollout, types
from imitation.util import logger
from imitation.util import sacred as sacred_util
from imitation.util import util

train_dagger_ex = sacred.Experiment("train_bc")


@train_dagger_ex.config
def config():
    env_name = "CartPole-v1"
    env = None
    n_epochs = None
    n_batches = None
    policy_save_path = None
    n_episodes_eval = 50
    expert_data = None
    rollout_hint = None
    n_expert_demos = None
    l2_weight = 3e-5
    batch_size = 512  # default 32
    expert_data = None
    expert_policy = "tests/data/expert_models/cartpole_0/policies/final"

    total_timesteps = 1e5

    log_root = pathlib.Path("output", "train_dagger")  # output directory
    optimizer_kwargs = dict(
        lr=4e-4,
    )  # TODO(shwang): Move config shared with BC into a `bc` Ingredient.


@train_dagger_ex.config
def defaults(
    env_name,
    env,
    rollout_hint,
):
    # TODO(shwang): Move {expert_data,expert_policy}_path to shared Ingredient as well.
    if rollout_hint is not None:
        expert_data = (
            f"data/expert_models/{rollout_hint}_0/rollouts/final.pkl"
        )
        expert_policy = (
            f"data/expert_models/{rollout_hint}_0/policies/final"
        )

    if env_name is not None and env is None:
        env = gym.make(env_name)


@train_dagger_ex.config
def default_train_duration(n_epochs, n_batches):
    if n_epochs is None and n_batches is None:
        n_epochs = 400


@train_dagger_ex.config
def paths(log_root, env_name):
    if env_name is None:
        _env_name_part = "unknown_env_name"
    else:
        _env_name_part = env_name.replace("/", "_")

    log_dir = pathlib.Path(log_root) / _env_name_part / util.make_unique_timestamp()
    del _env_name_part


# TODO(shwang): Move these redundant configs into a `auto.env` Ingredient,
# similar to what the ILR project does.
@train_dagger_ex.named_config
def mountain_car():
    env_name = "MountainCar-v0"
    rollout_hint = "mountain_car"


@train_dagger_ex.named_config
def seals_mountain_car():
    env_name = "seals/MountainCar-v0"
    rollout_hint = "seals_mountain_car"


@train_dagger_ex.named_config
def cartpole():
    env_name = "CartPole-v1"
    rollout_hint = "cartpole"


@train_dagger_ex.named_config
def seals_cartpole():
    env_name = "seals/CartPole-v0"
    rollout_hint = "cartpole"


@train_dagger_ex.named_config
def pendulum():
    env_name = "Pendulum-v0"
    rollout_hint = "pendulum"


@train_dagger_ex.named_config
def ant():
    env_name = "Ant-v2"
    rollout_hint = "ant"


@train_dagger_ex.named_config
def half_cheetah():
    env_name = "HalfCheetah-v2"
    rollout_hint = "half_cheetah"
    # Consider adding callback so that I can calculate reward over time.


@train_dagger_ex.named_config
def humanoid():
    env_name = "Humanoid-v2"
    rollout_hint = "humanoid"


@train_dagger_ex.named_config
def fast():
    n_batches = 50
    n_episodes_eval = 1
    n_expert_demos = 1


@train_dagger_ex.main
def train_dagger(
    _run,
    total_timesteps: float,
    log_dir: types.AnyPath,
    env: gym.Env,
    batch_size: int,
    expert_data: Union[types.AnyPath, types.Trajectory],
    expert_policy: Union[types.AnyPath, policies.BasePolicy],
    n_epochs: Optional[int],
    n_batches: Optional[int],
    n_episodes_eval: int,
    n_expert_demos: int,
    l2_weight: float,
    optimizer_kwargs: dict,
) -> dict:
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Logging to %s", log_dir)
    assert expert_policy is not None

    logger.configure(log_dir, ["tensorboard", "stdout"])
    sacred_util.build_sacred_symlink(log_dir, _run)

    if not isinstance(expert_policy, policies.BasePolicy):
        # Oof, we can't load the vec normalize without running DAgger with VecEnv.
        expert_policy = th.load(expert_policy)
    assert isinstance(expert_policy, policies.BasePolicy)

    if isinstance(expert_data, (str, pathlib.Path)):
        expert_trajs = types.load(expert_data)
    else:  # List of trajectories.
        expert_trajs = expert_data
    for x in expert_trajs:
        assert isinstance(x, types.TransitionsMinimal)

    # Copied from scripts/train_adversarial -- refactor with "auto"?
    if n_expert_demos is not None:
        if not len(expert_trajs) >= n_expert_demos:
            raise ValueError(
                f"Want to use n_expert_demos={n_expert_demos} trajectories, but only "
                f"{len(expert_trajs)} are available."
            )
        expert_trajs = expert_trajs[:n_expert_demos]

    model = dagger.SimpleDAggerTrainer(
        env=env,
        log_dir=log_dir,
        expert_trajs=expert_trajs,
        expert_policy=expert_policy,
        bc_kwargs=dict(
            l2_weight=l2_weight,
            optimizer_kwargs=optimizer_kwargs,
            batch_size=batch_size,
        ),
    )
    model.train(
        total_timesteps=int(total_timesteps),
        bc_n_epochs_per_round=n_epochs,
        bc_n_batches_per_round=n_batches,
    )
    save_info_dict = model.save_trainer()
    print(f"Model saved to {save_info_dict}")
    print(f"Tensorboard command: tbl '{log_dir}'")

    sample_until = rollout.make_sample_until(
        n_timesteps=None, n_episodes=n_episodes_eval
    )
    venv = vec_env.DummyVecEnv([lambda: env])
    trajs = rollout.generate_trajectories(
        model.policy,
        venv,
        sample_until=sample_until,
    )
    results = {}
    results["expert_stats"] = rollout.rollout_stats(expert_trajs)
    results["imit_stats"] = rollout.rollout_stats(trajs)
    return results


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_dagger"))
    train_dagger_ex.observers.append(observer)
    train_dagger_ex.run_commandline()


if __name__ == "__main__":
    main_console()
