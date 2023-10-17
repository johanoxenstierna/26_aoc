
import numpy as np
import os
import json

'''fixing profiles'''
with open('./profiles.json', 'r') as f:
	profiles = json.load(f)

for p_name, p in profiles.items():
	p['ELO_date'] = [[p['ELO'], 2309150000]]

with open('profiles.json', 'w') as f:
	json.dump(profiles, f, indent=1)