


"""
Attempts to simplify RFs problem by reducing number of features by 2

    0     1          2                3               4                5                6              7            8             9              10               11              12                13          14              15
winner,  ELO0 , ini_actions_prop0, ini_objs0, ini_objs_prop0, ini_targets_prop0, ini_group_size_avg0, ELO1, ini_actions_prop1, ini_objs1, ini_objs_prop1, ini_targets_prop1, ini_group_size_avg1, time_cut, profile_id_save   match_time


won_lost
ELO
p['ini_actions_prop'] = 0  # THE LARGER THE MORE INI
p['ini_objs'] = 0  # THE LARGER THE MORE INI
p['ini_objs_prop'] = 0  # THE LARGER THE MORE INI
p['ini_targets_prop'] = 0  # THE LARGER THE MORE INI
p['ini_group_size_avg'] = 0  # need to remove later if 0
time_cut
"""

import numpy as np

from src.analysis_utils import *

PATH_IN = './data_proc/D50.npy'
PATH_OUT = './data_proc/D50_diffs.npy'

D = np.load(PATH_IN)
D = D[np.where((D[:, 1] > 0) & (D[:, 7] > 0))[0], :]  # remove bad rows

# [ELO, ini_actions_prop, ini_objs, ini_objs_prop, ini_targets_prop, ini_group_size_avg]
WIN_COLS = [1, 2, 3, 4, 5, 6]
LOSS_COLS = [7, 8, 9, 10, 11, 12]
COLS = WIN_COLS[1:] + LOSS_COLS[1:]

D_ = weighted_means(D, COLS)  # this function doesnt care about winner-loser
D_flat = flatten_winner_loser(D_, TIME_CUT=1.0)

ELO_diff = D_flat[:, 1] - D_flat[:, 7]
ELO_diff = min_max_normalization(ELO_diff, y_range=[-1, 1])
ini_actions_prop_diff = D_flat[:, 2] - D_flat[:, 8]
ini_objs_diff = D_flat[:, 3] - D_flat[:, 9]
ini_objs_prop_diff = D_flat[:, 4] - D_flat[:, 10]
ini_targets_prop_diff = D_flat[:, 5] - D_flat[:, 11]
ini_group_size_avg_diff = D_flat[:, 6] - D_flat[:, 12]

D_out = np.zeros(shape=(len(D_flat), 8))
D_out[:, 0] = D_flat[:, 0]
D_out[:, 1] = ELO_diff
D_out[:, 2] = ini_actions_prop_diff
D_out[:, 3] = ini_objs_diff
D_out[:, 4] = ini_objs_prop_diff
D_out[:, 5] = ini_targets_prop_diff
D_out[:, 6] = ini_group_size_avg_diff
D_out[:, 7] = D_flat[:, 13]

np.save(PATH_OUT, D_out)


adf = 4


