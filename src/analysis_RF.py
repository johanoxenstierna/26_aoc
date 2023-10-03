

import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt

"""
0         1          2                  3            4                5                6        7                   8             9                 10                11
winner,  ELO0 , ini_times_avg_rat0, ini_objs_tot0, ini_targets0, ini_group_size_avg0, ELO1, ini_times_avg_rat1, ini_objs_tot1, ini_targets1, ini_group_size_avg1, profile_id
"""

D = np.load('./data_proc/D3_6000.npy')
# np.random.shuffle(D[:, 0])

'''Remove matches where diff in ELO is large'''
rows_to_keep = []
for i in range(0, len(D)):
	diff_elo = abs(D[i, 1] - D[i, 6])
	if diff_elo < 20:
		rows_to_keep.append(i)
D = D[rows_to_keep, :]

'''Manual: rows where a player took initiative vs not'''
rows_winner_and_first = np.where()

y = pd.DataFrame({'win0': pd.Series(D[:, 0], dtype='bool')})

X = pd.DataFrame({
	# 'elo0': pd.Series(D[:, 1], dtype='int'),
	# 'ini_times_avg_rat0': pd.Series(D[:, 2], dtype='float'),
	# 'ini_objs_tot0': pd.Series(D[:, 3], dtype='int'),
	# 'ini_targets0': pd.Series(D[:, 4], dtype='int'),
	'elo1': pd.Series(D[:, 6], dtype='int'),
	# 'ini_times_avg_rat1': pd.Series(D[:, 7], dtype='float'),
	# 'ini_objs_tot1': pd.Series(D[:, 8], dtype='int'),
	# 'ini_targets1': pd.Series(D[:, 9], dtype='int')
	}
)

# X = pd.DataFrame({
# 	'elo0': pd.Series(D[:, 1], dtype='int'),
# 	'elo1': pd.Series(D[:, 6], dtype='int')}
# )  # 0.68

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

m = RandomForestClassifier(n_estimators=100, max_depth=5)
cv_results = cross_validate(m, X, y.values.ravel(), cv=10)
print("mean accuracy: " + str(np.mean(cv_results['test_score'])))

# m.fit(X_train, y_train.values.ravel())

# importances = m.feature_importances_
# std = np.std([tree.feature_importances_ for tree in m.estimators_], axis=0)
#
# y_pred = m.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("accuracy: " + str(accuracy))
#
# feature_names = list(X.columns)
# forest_importances = pd.Series(importances, index=feature_names)
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()

# feature_names = [
# 	'elo0',
# 	'ini_times_avg_rat0',
# 	'ini_objs_tot0',
# 	'ini_targets0',
# 	# 'elo1',
# 	# 'ini_times_avg_rat1',
# 	# 'ini_objs_tot1',
# 	# 'ini_targets1'
# ]


# plt.show()

