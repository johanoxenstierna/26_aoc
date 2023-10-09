
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# sns.reset_orig()
import pandas as pd

from src.analysis_utils import *

"""
0       1          2                  3              4                5                6
ELO,  won Y/N, ini_times_avg_rat, ini_objs_tot, ini_targets, ini_group_size_avg,   profile_id

0         1          2                  3            4                5                6        7                   8             9                 10                11
winner,  ELO0 , ini_times_avg_rat0, ini_objs_tot0, ini_targets0, ini_group_size_avg0, ELO1, ini_times_avg_rat1, ini_objs_tot1, ini_targets1, ini_group_size_avg1, profile_id

	0     1          2                  3            4                5                6        7                   8             9                 10                11      12                13
winner,  ELO0 , ini_times_avg_rat0, ini_objs_tot0, ini_targets0, ini_group_size_avg0, ELO1, ini_times_avg_rat1, ini_objs_tot1, ini_targets1, ini_group_size_avg1, time_cut, profile_id_save   match_time


"""

# D = np.load('./data_proc/DD3_6000.npy')
D = np.load('./data_proc/D4.npy')
D = D[np.where(D[:, 1] > 0)[0], :]

asdf = 5
# '''Remove matches where diff in ELO is large'''
# rows_to_keep = []
# for i in range(0, len(D)):
# 	diff_elo = abs(D[i, 1] - D[i, 6])
# 	if diff_elo < 20:
# 		rows_to_keep.append(i)
# D = D[rows_to_keep, :]

'''Matches where the player who was faster also won'''
num_true = 0
for i in range(0, len(D)):
	if D[i, 0] < 0.5 and D[i, 2] < D[i, 7]:
		num_true += 1
	elif D[i, 0] > 0.5 and D[i, 2] > D[i, 7]:
		num_true += 1

win_rate = num_true / len(D)

aa = 5
# COL = 2
# D = D[np.where((D[:, 0] > 0) & (D[:, COL] > 0))[0], :]  # only rows with ELO and good COL

# win_rows = np.where(D[:, 1] > 0.5)[0]
# losses_rows = np.where(D[:, 1] < 0.5)[0]
#
# win_and_fix_rows = np.where((D[:, 1] > 0.5) & (D[:, 2] < 0.34))[0]
# loss_and_fix_rows = np.where((D[:, 1] < 0.5) & (D[:, 2] < 0.34))[0]

# wins_and_rat = np.mean(D[win_rows, 2])
# losses_and_rat = np.mean(D[losses_rows, 2])


afd = 5


