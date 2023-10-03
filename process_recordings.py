

import os
import json
import time

from mgz_aoc_clone.mgz import header, fast
from mgz_aoc_clone.mgz.summary import Summary
from mgz_aoc_clone.mgz.model import parse_match, serialize

from src.utils import *

# PATH = './mgz_aoc_clone/tests/recs_first/'
# PATH = './r/first/'
PATH = './r/3/'
# PATH_REC = 'r/TRM_Pyroptere.aoe2record'
PATH_REC = 'r/test.aoe2record'
OUT_NAME = './data_proc/D.npy'

# with open(PATH_REC, 'rb') as f:
#     s = Summary(f)
#     aa = s.get_players()  # INCLUDES EAPM

# from mgz_aoc_clone.mgz import header, fast
#
# with open(PATH_REC, 'rb') as data:
#     eof = os.fstat(data.fileno()).st_size
#     header.parse_stream(data)
#     fast.meta(data)
#     while data.tell() < eof:
#         fast.operation(data)

adf = 5
_, _, file_names = os.walk(PATH).__next__()
LEN_DATA = len(file_names)
D = np.zeros(shape=(LEN_DATA * 2, 7), dtype=float)
cur_row = 0
time0 = time.time()  # 4000 s to do 6000 files

with open('./profiles.json', 'r') as f:
    profiles = json.load(f)

files_to_be_removed = []

ii = 0
for i in range(ii, len(file_names)):

    file_name = file_names[i]
    # file_name = '1088465_2309260214.aoe2record'

    try:
        profile_id = int(file_name.split('_')[0])
    except:
        profile_id = 999999

    # match_time = file_name.split('_')[1]  # not used

    print("ii: " + str(ii) + "  " + str(file_name))

    with open(PATH + file_name, 'rb') as f:
        m, ps = parse_match(f)

    if len(m.players) > 2:
        print("!!!More than two PLAYERS")

    flag_prof_not_found = get_profiles(profile_id, ps, profiles)
    if flag_prof_not_found == True:
        continue

    if len(m.actions) < 500:
        print("Too few actions: " + str(file_name))
        files_to_be_removed.append(file_name)
        continue

    flag_not_found_loser = get_ps_actions(m.actions, ps)
    if flag_not_found_loser == True:
        print("flag_not_found_loser")
        files_to_be_removed.append(file_name)
        continue

    try:
        get_tc_coords(ps)
    except:
        print("get_tc_coords error")
        files_to_be_removed.append(file_name)
        continue

    flag_not_found_aggr = get_aggr_actions(ps)
    if flag_not_found_aggr == True:
        print("No aggr actions found for file_name: " + str(file_name))
        files_to_be_removed.append(file_name)
        continue

    get_early_initiative(ps)

    cur_row = infer_and_push_to_D(cur_row, D, ps, profile_id)

    ii += 1
    # if ii > 2000:
    #     break

with open("files_to_be_removed.json", 'w') as f:
    json.dump(files_to_be_removed, f, indent=2)

time1 = time.time() - time0
print("took " + str(time1) + " to do " + str(ii) + " games")
'''TODO: remove missing rows + print how many missing'''

D = D[np.where(D[:, 0] > 0)[0], :]
np.save(OUT_NAME, D)
sss = 4