
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

from dagger_experiment import train_dagger, make_trainer

TOTAL_STEPS = 30000
N_POINTS_TO_PLOT = 10

def run_rampdown_experiment():

    for point in range(N_POINTS_TO_PLOT):

        decay_end_point = TOTAL_STEPS // (point + 1)
        

    pass