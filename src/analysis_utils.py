
import numpy as np

def min_max_normalization(X, y_range):

	new_min = y_range[0]
	new_max = y_range[1]
	Y = np.zeros(X.shape)

	_min = np.min(X)
	_max = np.max(X)

	for i, x in enumerate(X):
		Y[i] = ((x - _min) / (_max - _min)) * (new_max - new_min) + new_min

	return Y


def convert_to_single_row(DD):
	"""temp: until process_recordings does this

	0       1          2                  3              4                5                6
	ELO,  won Y/N, ini_times_avg_rat, ini_objs_tot, ini_targets, ini_group_size_avg,   profile_id

	to

	0         1          2                  3            4                5                6        7                   8             9                 10                11
	winner,  ELO0 , ini_times_avg_rat0, ini_objs_tot0, ini_targets0, ini_group_size_avg0, ELO1, ini_times_avg_rat1, ini_objs_tot1, ini_targets1, ini_group_size_avg1, profile_id

	"""

	'''
	first the rows with time > 0.3333 need to be removed. THEY ARE INVALID -> the assumption is that 
	we are only aware of what happens at the first third of aggression
	'''

	D = np.zeros(shape=(len(DD), 12), dtype=float)

	i = 0  # D
	i0 = 0  # DD
	i1 = 1  # DD
	while i1 < len(DD):
		row0 = DD[i0, :]
		row1 = DD[i1, :]

		if (row0[6] != row1[6]):
			print("no pair game")
			i0 += 1
			i1 += 1
			continue

		if (row0[0] < 10 or row1[0] < 10):
			print("player missing elo")
			i0 += 2
			i1 += 2
			continue

		if row0[1] > 0.5 and row1[1] < 0.5:
			D[i, 0] = 0
		elif row0[1] < 0.5 and row1[1] > 0.5:
			D[i, 0] = 1
		else:
			raise Exception("wrong winner thing")

		D[i, 1] = row0[0]  # elo
		D[i, 6] = row1[0]  # elo

		if row0[2] < 0.34:  # ini_times_avg_rat
			D[i, [2, 3, 4, 5]] = [row0[2], row0[3], row0[4], row0[5]]
		else:  # restore defaults
			D[i, [2, 3, 4, 5]] = [1, 0, 0, 0]

		if row1[2] < 0.34:  # ini_times_avg_rat
			D[i, [7, 8, 9, 10]] = [row1[2], row1[3], row1[4], row1[5]]
		else:
			D[i, [7, 8, 9, 10]] = [1, 0, 0, 0]

		D[i, 11] = row0[6]

		i += 1  # D
		i0 += 2  # DD
		i1 += 2  # DD

	D = D[np.where(D[:, 1] > 0)[0], :] # needed to remove the extra rows

	np.save('./data_proc/D3_6000.npy', D)
