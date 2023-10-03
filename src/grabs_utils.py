import copy
import io
import random
import zipfile
import requests
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from uuid import uuid4
import datetime
import json
import numpy as np


def get_times(driver, trs):

	match_times = []
	fails = 0

	for i, tr in enumerate(trs):

		driver.execute_script("window.scrollTo(0, 500)")
		time.sleep(0.1)
		driver.execute_script("window.scrollTo(0, -500)")
		time.sleep(0.1)

		_t = ""
		_s = ""
		out = ""

		try:
			_t = tr.text
			_s = _t.split('\n')

			d = datetime.datetime.strptime(_s[0], '%B %d, %Y at %H:%M %p')
			out = d.strftime('%y%m%d%H%M')
			match_times.append(out)

		except:
			print("could not get match time: _t: " + str(_t) + " _s: " + str(_s) + " out: " + str(out))
			rand_ = random.randint(0, 99999999)
			s_rand_ = "{:08d}".format(rand_)
			match_times.append('99' + s_rand_)
			fails += 1

	print("parsed times: " + str(len(match_times)) + " successful. " + str(fails) + " failures.")

	return match_times


def get_profile_ids(num, out_names_done):

	out_profile_ids_done = [int(x.split('_')[0]) for x in out_names_done]

	with open('./profiles.json', 'r') as f:
		profiles = json.load(f)

	profile_names = list(profiles.keys())

	# names = list(np.random.choice(np.asarray(profile_ids), size=[num,]))
	# names = list(np.random.choice(np.asarray(profile_ids), size=[num,]))
	random.shuffle(profile_names)

	# num0 = int(num / 2)
	# num1 = 200

	selection = []

	for name in profile_names:
		if profiles[name]['profile_id'] not in out_profile_ids_done:
			selection.append(profiles[name]['profile_id'])

			if len(selection) >= num:
				break

	# for name in profile_names:
	# 	# if name in selection_names:
	# 	if profiles[name]['ELO'] < 1300:
	# 		selection.append(profiles[name]['profile_id'])
	#
	# 	if len(selection) >= num0:
	# 		print("last profile name: " + str(profiles[name]) + "  ELO: " + str(profiles[name]['ELO']))
	# 		break

	# for name in profile_names:
	# 	# if name in selection_names:
	# 	if profiles[name]['ELO'] > 1800:
	# 		selection.append(profiles[name]['profile_id'])
	#
	# 	if len(selection) >= num1:
	# 		print("last profile name: " + str(profiles[name]) + "  ELO: " + str(profiles[name]['ELO']))
	# 		break

	return selection


def profiles_postproc():

	"""
	manual additions: ... forgot
	:return:
	"""

	def is_jsonable(x):
		try:
			json.dumps(x)
			return True
		except:  # (TypeError, OverflowError)
			return False

	with open('./profiles.json', 'r') as f:  # 3961 ELOS, lowest one: 1400
		profiles = json.load(f)

	# elos_profileids = np.zeros(shape=(len(profiles), 2), dtype=int)
	#
	# elos_profileids[0, 1] = 44545
	#
	# profile_ids = list(profiles.keys())
	#
	# for i in range(len(profile_ids)):
	# 	profile = profiles[profile_ids[i]]
	# 	elos_profileids[i, 0] = profile['ELO']
	# 	elos_profileids[i, 1] = profile['profile_id']

	# np.save('./elos_profileids.npy', elos_profileids)

	profiles_out = {}
	for i, profile in profiles.items():

		profile_out = {
			'name': str(profile['name']),
			'profile_id': profile['profile_id'],
			'ELO': profile['ELO']}

		result = is_jsonable(profile_out)

		if result == True:
			profiles_out[profile['name']] = profile
		else:
			print("could not save name: " + str(profile['name']))

		ad = 4

	with open('./profiles.json', 'w') as f:
		json.dump(profiles_out, f, indent=2)

