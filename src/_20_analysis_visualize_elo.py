
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# sns.reset_orig()
import pandas as pd

from src.analysis_utils import *

D = np.load('./data_proc/D20.npy')  # OBS DONT LOOK AT LEN HERE
D = D[np.where((D[:, 1] > 0) & (D[:, 7] > 0))[0], :]

TIME_CUT = 0.5
D = D[np.where(D[:, 13] < TIME_CUT)[0], :]

'''Keep matches where diff in ELO is low'''
rows_to_keep = []
for i in range(0, len(D)):
    diff_elo = abs(D[i, 1] - D[i, 7])
    if diff_elo < 200:
        rows_to_keep.append(i)

print("Rows before: " + str(len(D)) + "  Rows aft: " + str(len(rows_to_keep)))
D = D[rows_to_keep, :]   # break here to see how many were kept

# D = convert_to_single_row(DD)

"""
    0     1          2                3               4                5                6              7            8             9              10               11              12                13          14              15
winner,  ELO0 , ini_actions_prop0, ini_objs0, ini_objs_prop0, ini_targets_prop0, ini_group_size_avg0, ELO1, ini_actions_prop1, ini_objs1, ini_objs_prop1, ini_targets_prop1, ini_group_size_avg1, time_cut, profile_id_save   match_time

PLACEHOLDER (win/loss)
ELO
p['ini_actions_prop'] = 0  # THE LARGER THE MORE INI
p['ini_objs'] = 0  # THE LARGER THE MORE INI
p['ini_objs_prop'] = 0  # THE LARGER THE MORE INI
p['ini_targets_prop'] = 0  # THE LARGER THE MORE INI
p['ini_group_size_avg'] = 0  # need to remove later if 0

OBS can only do 1 column, BUT DOES NOT MATTER CUZ LOOK AT V 
"""

# [ELO, ini_actions_prop, ini_objs, ini_objs_prop, ini_targets_prop, ini_group_size_avg]
# COLS = [0, 1, 2, 3, 4, 5]
COL_TO_TEST_INDEX = 4  # THIS IS AN INDEX!!!
WIN_COLS = [1, 2, 3, 4, 5, 6]
LOSS_COLS = [7, 8, 9, 10, 11, 12]

# aa = min_max_normalization(D[:, 3], y_range=[-1, 0])

# win_rows = np.where(D[:, 1] > 0.5)[0]
# losses_rows = np.where(D[:, 1] < 0.5)[0]

# wins_and_rat = np.mean(D[win_rows, 2])
# wins_and_1_rat = np.where((D[:, 1] > 0.5) & (np.isclose(D[:, 2], 1.0)))[0]
# losses_and_rat = np.mean(D[losses_rows, 2])

# first_rows = np.where((D[:, COL + 3] > 0) & (D[:, COL + 3] < 10))[0]
# secs_rows = np.where((D[:, COL + 3] < 1) & (D[:, COL + 3] < 10))[0]

# wins = D[win_rows, :]
# losses = D[losses_rows, :]
min_ = 0
# all_ok_rows_winner = np.where(wins[:, COL] > min_)[0]
# all_ok_rows_loser = np.where(losses[:, COL] > min_)[0]

# wins_and_first = np.where((wins[:, COL + 3] > 0.5) & (wins[:, COL] > min_))[0]
# wins_and_sec = np.where((wins[:, COL + 3] < 0.5) & (wins[:, COL] > min_))[0]

# losses_and_first = np.where((losses[:, COL + 3] > 0.5) & (losses[:, COL] > min_))[0]
# losses_and_sec = np.where((losses[:, COL + 3] < 0.5) & (losses[:, COL] > min_))[0]

# firsts = D[first_rows, :]
# secs = D[secs_rows, :]

'''some stats'''
won_and_COL = np.mean(D[:, WIN_COLS[COL_TO_TEST_INDEX]])
loss_and_COL = np.mean(D[:, LOSS_COLS[COL_TO_TEST_INDEX]])

print("won_and_COL: " + str(won_and_COL))
print("lost_and_COL: " + str(loss_and_COL))

# won_and_ELO = np.mean(wins[:, 0])
# lost_and_ELO = np.mean(losses[:, 0])
# gradient = 1 - lost_and_ELO / won_and_ELO
# print("gradient ELO diff: " + str(gradient))


''' cols using time'''
# D = wins

# wins_avg_time = np.mean(wins[wins_and_first, COL])
# losses_avg_time = np.mean(losses[losses_and_sec, COL])

fig, ax0 = plt.subplots(figsize=(12, 12))

'''
Scatter plot and reg line
DEPR CUZ IT ONLY DOES 1 ELO
'''
# temp = np.zeros(shape=(len(D), 2), dtype=int)
# temp[:, 0], temp[:, 1] = D[:, WIN_COLS[0]], D[:, WIN_COLS[COL_TO_TEST_INDEX]]
# df_blue = pd.DataFrame(temp, columns=['ELO', 'COL'])
# ax_blue = ax0.scatter(df_blue['ELO'], df_blue['COL'], c='blue', s=100, alpha=0.1)

# # res = stats.goodness_of_fit(stats.norm, A, statistic='ks', random_state=np.random.default_rng())
# slope_blue, intercept, r_value, p_value_blue, std_err = stats.linregress(wins[all_ok_rows_winner, 0], wins[all_ok_rows_winner, COL])
# slope_red, intercept, r_value, p_value_red, std_err = stats.linregress(losses[all_ok_rows_loser, 0], losses[all_ok_rows_loser, COL])
# print("slope_blue: " + str(slope_blue) + "  p_value blue: " + str(p_value_blue))
# print("slope_loser: " + str(slope_red) + "  p_value red: " + str(p_value_red))
#
# ax_blue_reg = sns.regplot(x=df_blue['ELO'], y=df_blue['time'], ci=True, line_kws={'color': 'blue', 'alpha': 0.3}, scatter=False)
# ax_red_reg = sns.regplot(x=df_red['ELO'], y=df_red['time'], ci=True, line_kws={'color': 'red', 'alpha': 0.3}, scatter=False)


'''
violin
Needs the winner and loss data to be stacked
'''

# [elo_cat, won_lost, COL_TO_TEST value]
V = np.zeros(shape=(len(D) * 2, 3), dtype=np.float32)  # the input to the violin plot
win_rows = np.arange(0, len(D))
loss_rows = np.arange(len(D), len(D) * 2)

V[win_rows, 1] = 1
V[loss_rows, 1] = 0

V[win_rows, 2] = D[:, WIN_COLS[COL_TO_TEST_INDEX]]
V[loss_rows, 2] = D[:, LOSS_COLS[COL_TO_TEST_INDEX]]

elos = np.zeros(shape=(len(V),), dtype=np.float32)
elos[win_rows] = D[:, WIN_COLS[0]]
elos[loss_rows] = D[:, LOSS_COLS[0]]

# to_match = np.linspace(1000, 2900, 10, dtype=int)
to_match = list(range(900, 3100, 300))
# avg_elos = np.zeros(shape=(len(D),), dtype=float)
#
for i in range(0, len(to_match) - 1):
    matches = np.where((elos >= to_match[i]) & (elos < to_match[i + 1]))[0]
    V[matches, 0] = to_match[i] + 100


# elo_cats = np.zeros(shape=(len(D),), dtype=int)

# for i in range(0, len(to_match) - 1):
#     # matches = np.where((avg_elos >= to_match[i]) & (avg_elos < to_match[i + 1]))[0]
#     matches = np.where((avg_elos >= to_match[i]) & (avg_elos < to_match[i + 1]))[0]
#     elo_cats[matches] = to_match[i] + 100

# D = np.concatenate((D, elo_cats), axis=1)
# D = D[np.where(D[:, -1] > 10)[0], :]
# D = D[0:500, :]

# temp = D[:, [1, 12, COL]].astype(int)
# df = pd.DataFrame(temp, columns=['won_lost', 'elo_cats', 'COL'])
df = pd.DataFrame({'Won?': pd.Series(V[:, 1], dtype='bool'),
                   'elo_cats': pd.Series(V[:, 0], dtype='int'),
                   'COL': pd.Series(V[:, 2], dtype='float')})

# sns.violinplot(data=df, x="COL", y="elo_cats", hue="Won?", split=True, orient='h',
#                # hue_order=[True, False],
#                palette={True: 'blue', False: 'red'},
#                cut=0,
#                # inner=None,
#                density_norm='count'
#                )

sns.violinplot(data=df, x="elo_cats", y="COL", hue="Won?", split=True, orient='v',
               # hue_order=[True, False],
               palette={True: 'blue', False: 'red'},
               cut=0,
               density_norm='area',
               # inner=None
               )

# plt.gca().invert_yaxis()  # DOESNT WORK FOR INNER


plt.show()
adf = 5