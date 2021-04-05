import logging
import os.path as osp
import pathlib
import pickle
from typing import Iterable, Mapping, Optional, Union

import gym
import sacred
from gym import spaces
from sacred.observers import FileStorageObserver
from stable_baselines3.common import vec_env
from torch.utils import data as th_data

from imitation.algorithms import bc
from imitation.data import rollout, types
from imitation.util import logger
from imitation.util import sacred as sacred_util
from imitation.util import util

train_bc_ex = sacred.Experiment("train_bc")


@train_bc_ex.config
def config():
    env_name = "CartPole-v1"
    venv = None
    observation_space = None
    action_space = None
    expert_data_type = ""
    n_epochs = None
    n_batches = None
    policy_save_path = None
    n_episodes_eval = 50
    expert_data_src = None
    expert_data_src_format = None
    rollout_hint = None
    n_expert_demos = None
    l2_weight = 3e-5
    batch_size = 512  # default 32

    log_root = pathlib.Path("output", "train_bc")  # output directory
    optimizer_kwargs = dict(
        lr=4e-4,
    )


# Learning rate too high?
# Dataset too big?
# Neural network too small?


@train_bc_ex.config
def defaults(
    expert_data_src,
    expert_data_src_format,
    env_name,
    venv,
    observation_space,
    action_space,
    rollout_hint,
):
    if expert_data_src is None and expert_data_src_format is None:
        expert_data_src = (
            "data/expert_models/" f"{rollout_hint or 'cartpole'}_0/rollouts/final.pkl"
        )
        expert_data_src_format = "path"

    if env_name is not None:
        venv = util.make_vec_env(env_name)  # TODO(shwang): Use auto in the future...

    if venv is not None:
        if observation_space is None:
            observation_space = venv.observation_space

        if action_space is None:
            action_space = venv.action_space


@train_bc_ex.config
def default_train_duration(n_epochs, n_batches):
    if n_epochs is None and n_batches is None:
        n_epochs = 400


@train_bc_ex.config
def paths(log_root, env_name):
    if env_name is None:
        _env_name_part = "unknown_env_name"
    else:
        _env_name_part = env_name.replace("/", "_")

    log_dir = pathlib.Path(log_root) / _env_name_part / util.make_unique_timestamp()
    del _env_name_part


@train_bc_ex.named_config
def mountain_car():
    env_name = "MountainCar-v0"
    rollout_hint = "mountain_car"


@train_bc_ex.named_config
def seals_mountain_car():
    env_name = "seals/MountainCar-v0"
    rollout_hint = "seals_mountain_car"


@train_bc_ex.named_config
def cartpole():
    env_name = "CartPole-v1"
    rollout_hint = "cartpole"


@train_bc_ex.named_config
def seals_cartpole():
    env_name = "seals/CartPole-v0"
    rollout_hint = "cartpole"


@train_bc_ex.named_config
def pendulum():
    env_name = "Pendulum-v0"
    rollout_hint = "pendulum"


@train_bc_ex.named_config
def ant():
    env_name = "Ant-v2"
    rollout_hint = "ant"


@train_bc_ex.named_config
def half_cheetah():
    env_name = "HalfCheetah-v2"
    rollout_hint = "half_cheetah"
    # Consider adding callback so that I can calculate reward over time.


@train_bc_ex.named_config
def humanoid():
    env_name = "Humanoid-v2"
    rollout_hint = "humanoid"


@train_bc_ex.named_config
def fast():
    n_batches = 50
    n_episodes_eval = 1
    n_expert_demos = 1


@train_bc_ex.main
def train_bc(
    _run,
    log_dir: types.AnyPath,
    env_name: str,  # Probably use some auto ingredient instead next time.
    venv: Optional[vec_env.VecEnv],
    batch_size: int,
    observation_space: gym.Space,
    action_space: gym.Space,
    # TODO(shwang): Doesn't currently accept Iterable[Mapping] or types.TransitionsMinimal.
    expert_data_src: Union[types.AnyPath, types.Trajectory],
    expert_data_src_format: str,
    n_epochs: Optional[int],
    n_batches: Optional[int],
    policy_save_path: Optional[str],
    n_episodes_eval: int,
    n_expert_demos: int,
    l2_weight: float,
    optimizer_kwargs: dict,
) -> dict:
    assert action_space is not None
    assert observation_space is not None

    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Logging to %s", log_dir)

    logger.configure(log_dir, ["tensorboard", "stdout"])
    sacred_util.build_sacred_symlink(log_dir, _run)

    assert expert_data_src_format in ("path", "trajectory")
    if expert_data_src_format == "path":
        expert_trajs = types.load(expert_data_src)
        # TODO(shwang): Convert the test data first, which is still compat-mode
        # with open(expert_data_src, "rb") as f:
        #     expert_data_trajs = pickle.load(f)
    elif expert_data_src_format == "trajectory":
        expert_trajs = expert_data_src
    else:
        raise ValueError(f"Invalid expert_data_src_format={expert_data_src_format}")

    # Copied from scripts/train_adversarial -- refactor with "auto"?
    if n_expert_demos is not None:
        if not len(expert_trajs) >= n_expert_demos:
            raise ValueError(
                f"Want to use n_expert_demos={n_expert_demos} trajectories, but only "
                f"{len(expert_trajs)} are available."
            )
        expert_trajs = expert_trajs[:n_expert_demos]

    expert_data_trans = rollout.flatten_trajectories(expert_trajs)
    expert_data = th_data.DataLoader(
        expert_data_trans,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=types.transitions_collate_fn,
    )

    model = bc.BC(
        observation_space,
        action_space,
        expert_data=expert_data,
        l2_weight=l2_weight,
        optimizer_kwargs=optimizer_kwargs,
    )
    model.train(n_epochs=n_epochs, n_batches=n_batches, rollout_venv=venv)
    if policy_save_path is not None:
        model.save_policy(policy_save_path)

    print(f"Tensorboard command: tbl '{log_dir}'")

    # Later: Use auto env, auto stats thing with shared `env` and stats ingredient,
    # or something like that.
    sample_until = rollout.make_sample_until(
        n_timesteps=None, n_episodes=n_episodes_eval
    )
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
    observer = FileStorageObserver(osp.join("output", "sacred", "train_bc"))
    train_bc_ex.observers.append(observer)
    train_bc_ex.run_commandline()


if __name__ == "__main__":
    main_console()
