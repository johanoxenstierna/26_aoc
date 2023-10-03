
import numpy as np

_1 = np.load('./data_proc/D1.npy')
_2 = np.load('./data_proc/D2.npy')
_3 = np.load('./data_proc/D3_12000.npy')

# A = A[np.where(A[:, 0] > 0)[0]]  # already done

# fix_A = np.zeros(shape=(len(A), 1))
# A = np.concatenate((A, fix_A), axis=1)
A = np.concatenate((_1, _2))
B = np.concatenate((A, _3))

# A = np.load('./data_proc/D3.npy')
# rows = np.where(A[:, 0] > 0)[0]
# A = A[rows, :]
np.save('./data_proc/D3_comb.npy', B)  #  OBBBBS

asdf = 5