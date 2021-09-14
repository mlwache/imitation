import numpy as np
import matplotlib.pyplot as plt

filename = 'test1.npz'

npzfile = np.load(filename)

episode_lengths = npzfile['episode_lengths']
times_failed = npzfile['times_failed']
failed = npzfile['failed']

TOTAL_STEPS = npzfile['TOTAL_STEPS']
N_POINTS_TO_PLOT = npzfile['N_POINTS_TO_PLOT']
MAX_STEPS_PER_TRAJECTORY = npzfile['MAX_STEPS_PER_TRAJECTORY']
N_TRAJECTORIES_PER_ROUND = npzfile['N_TRAJECTORIES_PER_ROUND']

decay_end_points = TOTAL_STEPS // (np.arange(N_POINTS_TO_PLOT) + 1)

plt.plot(decay_end_points, times_failed)

filename = 'test2_gets_knocked.npz'

npzfile = np.load(filename)

episode_lengths = npzfile['episode_lengths']
times_failed = npzfile['times_failed']
failed = npzfile['failed']

TOTAL_STEPS = npzfile['TOTAL_STEPS']
N_POINTS_TO_PLOT = npzfile['N_POINTS_TO_PLOT']
MAX_STEPS_PER_TRAJECTORY = npzfile['MAX_STEPS_PER_TRAJECTORY']
N_TRAJECTORIES_PER_ROUND = npzfile['N_TRAJECTORIES_PER_ROUND']

decay_end_points = TOTAL_STEPS // (np.arange(N_POINTS_TO_PLOT) + 1)

plt.plot(decay_end_points, times_failed)




plt.title(f'Total Steps: {TOTAL_STEPS}')
plt.ylabel('Failures')
plt.xlabel('Rampdown Steps')
plt.show()