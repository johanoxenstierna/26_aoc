
import numpy as np
import os
import json

A = np.array([[4, 5, 7, 2],
              [6, 3, 6, 2]], dtype=float)
# A[:, 0] += np.random.random(size=len(A[:, 0]))
A[:, 0] += np.random.uniform(low=0.5, high=13.3, size=len(A[:, 0]))

adf = np.random.uniform(low=0.5, high=13.3, size=(50,))

adf = 6
# os.system('mv ./r_test/2_z/1_ee07fb00.aoe2record.zip /media/johan/0E45-EEA5/r_z/')

# '''fixing profiles'''
# with open('./profiles.json', 'r') as f:
# 	profiles = json.load(f)
#
# for p_name, p in profiles.items():
# 	p['ELO_date'] = [[p['ELO'], 2309150000]]
#
# with open('profiles.json', 'w') as f:
# 	json.dump(profiles, f, indent=1)