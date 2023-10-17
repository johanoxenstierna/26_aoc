

import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt
from src.analysis_utils import *

"""
    0     1          2                  3                  4                5                6        7                   8             9                    10               11      12                13
winner,  ELO0 , ini_actions_prop0, ini_objs_prop0, ini_targets_prop0, ini_group_size_avg0, ELO1, ini_actions_prop1, ini_objs_prop1, ini_targets_prop1, ini_group_size_avg1, time_cut, profile_id_save   match_time

    0     1          2                3               4                5                6              7            8             9              10               11              12                13          14              15
winner,  ELO0 , ini_actions_prop0, ini_objs0, ini_objs_prop0, ini_targets_prop0, ini_group_size_avg0, ELO1, ini_actions_prop1, ini_objs1, ini_objs_prop1, ini_targets_prop1, ini_group_size_avg1, time_cut, profile_id_save   match_time

ELO
p['ini_actions_prop'] = 0  # THE LARGER THE MORE INI
p['ini_objs'] = 0  # THE LARGER THE MORE INI
p['ini_objs_prop'] = 0  # THE LARGER THE MORE INI
p['ini_targets_prop'] = 0  # THE LARGER THE MORE INI
p['ini_group_size_avg'] = 0  # need to remove later if 0


0 won_lost
1 ELO_diff
2 p['ini_actions_prop_diff'] = 0  # THE LARGER THE MORE INI
3 p['ini_objs_diff'] = 0  # THE LARGER THE MORE INI
4 p['ini_objs_prop_diff'] = 0  # THE LARGER THE MORE INI
5 p['ini_targets_prop_diff'] = 0  # THE LARGER THE MORE INI
6 p['ini_group_size_avg_diff'] = 0  # need to remove later if 0
7 time_cut

"""

D = np.load('./data_proc/D60_diffs_comb.npy')
# D = D[np.where(D[:, 1] > 0)[0], :]
# np.random.shuffle(D[:, 0])

# '''Keep matches where diff in ELO is low'''
rows_to_keep = []
for i in range(0, len(D)):
	# diff_elo = abs(D[i, 1] - D[i, 7])
	diff_elo = abs(D[i, 1])
	if diff_elo < 0.06:
		rows_to_keep.append(i)

print("Rows before: " + str(len(D)) + "  Rows aft: " + str(len(rows_to_keep)))
D = D[rows_to_keep, :]   # break here to see how many were kept

# time_cut_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# time_cut_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
time_cut_ratios = [0.5]  # OBS TEMP TIME_CUT - 0.3 used as frame
for i in range(len(time_cut_ratios)):

	TIME_CUT = time_cut_ratios[i]
	# D_flat = flatten_winner_loser(D, TIME_CUT)  # NEEDED TO HIDE THE ANSWER
	rows = np.where((D[:, 7] > (TIME_CUT - 0.05)) & (D[:, 7] < (TIME_CUT + 0.05)))[0]
	D_t = D[rows, :]

	#PEND DEL
	# rows = np.where(D_flat[:, 11] > 0.95)[0]
	# D_flat = D_flat[rows, :]

	'''Manual: rows where a player took initiative vs not'''
	# rows_winner_and_first = np.where()

	y = pd.DataFrame({'win0': pd.Series(D_t[:, 0], dtype='bool')})

	'''NEW. Merged cols'''
	X = pd.DataFrame({
		# 'elo_diff': pd.Series(D_t[:, 1], dtype='float'),  # TODO ELO DIFFERENCE
		'ini_actions_prop_diff': pd.Series(D_t[:, 2], dtype='float'),
		'ini_objs_diff': pd.Series(D_t[:, 3], dtype='float'),
		'ini_objs_prop_diff': pd.Series(D_t[:, 4], dtype='float'),
		'ini_targets_prop_diff': pd.Series(D_t[:, 5], dtype='float'),
		'ini_group_size_avg_diff': pd.Series(D_t[:, 6], dtype='float'),
		# 'time_cut': pd.Series(D_t[:, 7])  # completely useless for this, as it should be
	}
	)

	# X = pd.DataFrame({
	# 	# 'elo0': pd.Series(D_flat[:, 1], dtype='int'),  # TODO ELO DIFFERENCE
	# 	'ini_actions_prop0': pd.Series(D_t[:, 2], dtype='float'),
	# 	'ini_objs0': pd.Series(D_t[:, 3], dtype='int'),
	# 	'ini_objs_prop0': pd.Series(D_t[:, 4], dtype='float'),
	# 	'ini_targets_prop0': pd.Series(D_t[:, 5], dtype='float'),
	# 	# 'elo1': pd.Series(D_flat[:, 6], dtype='int'),
	# 	'ini_actions_prop1': pd.Series(D_t[:, 8], dtype='float'),
	# 	'ini_objs1': pd.Series(D_t[:, 9], dtype='int'),
	# 	'ini_objs_prop1': pd.Series(D_t[:, 10], dtype='float'),
	# 	'ini_targets_prop1': pd.Series(D_t[:, 11], dtype='float'),
	# 	'time_cut': pd.Series(D_t[:, 11])
	# 	}
	# )

	m = RandomForestClassifier(n_estimators=100, max_depth=10)

	'''No cv'''
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, shuffle=True)
	# m.fit(X_train, y_train.values.ravel())
	# y_pred = m.predict(X_test)
	# accuracy = accuracy_score(y_test, y_pred)
	# print("TIME_CUT: " + str(TIME_CUT) + " Mean accuracy: " + str(accuracy))

	'''cv'''
	cv_results = cross_validate(m, X, y.values.ravel(), cv=10, verbose=1)
	print("TIME_CUT: " + str(TIME_CUT) + " Mean accuracy: " + str(np.mean(cv_results['test_score'])))

	# # '''Feature importance'''
	# importances = m.feature_importances_
	# std = np.std([tree.feature_importances_ for tree in m.estimators_], axis=0)
	# feature_names = list(X.columns)
	# forest_importances = pd.Series(importances, index=feature_names)
	# fig, ax = plt.subplots()
	# forest_importances.plot.bar(yerr=std, ax=ax)
	# ax.set_title("Feature importances using MDI")
	# ax.set_ylabel("Mean decrease in impurity")
	# fig.tight_layout()
	# plt.show()
	# break

