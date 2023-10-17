
import os
import io
import shutil
import zipfile
import requests
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from uuid import uuid4

from src.grabs_utils import *


# Collection period: September 16 -

# profile_ids = [551102, 9701023, 2413974, 334309,
#                2007713, 10818645, 2071926, 586490, 403294, 11665616]

# problem_prof 2645439 2070169

PATH_OUT = './r/6_z/'
UNZIP_FOLDER = './r/unzip_folder/'
# PATH_OUT = '/media/johan/KINGSTON/r_z/'  # zip doesnt seem to work here
PATHS_DONE = ['./r/3/', './r/4/', './r/5_z/', './r/6_z/']  # '/media/johan/KINGSTON/r/'
COMPUTER_CUT = [0, 0.5]

'''TODO: fix out_names_done FOR ZIPS'''
_, _, out_names_done0 = os.walk(PATHS_DONE[0]).__next__()
_, _, out_names_done1 = os.walk(PATHS_DONE[1]).__next__()
_, _, out_names_done2 = os.walk(PATHS_DONE[2]).__next__()
_, _, out_names_done3 = os.walk(PATHS_DONE[3]).__next__()
out_names_done = out_names_done0 + out_names_done1 + out_names_done2 + out_names_done3
out_names_done = [x.split('.')[0] for x in out_names_done]
# out_names_done = []
profile_ids = get_profile_ids(3000, out_names_done, COMPUTER_CUT)

driver = webdriver.Firefox()
time0 = time.time()
num_games = 0
num_games_STOP = 14000
for iii, profile_id in enumerate(profile_ids):

    print("\n\n")
    print("==========================")
    print("profile_id: " + str(profile_id))
    print("==========================")
    print("Total games added: " + str(num_games))

    try:  # try a profile
        driver.get(f"https://www.ageofempires.com/stats/?profileId={profile_id}&game=age2&matchType=3")#put here the adress of your page
        wait = WebDriverWait(driver, timeout=60)  # log in
        wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "match-results__content")))

        # time.sleep(5)

        trs = driver.find_elements(by=By.CLASS_NAME, value="match-results__row")
        match_times = get_times(driver, trs)

        for i, tr in enumerate(trs):

            out_name = str(profile_id) + "_" + str(match_times[i])
            if out_name in out_names_done:
                print("file already done: out_name: " + str(out_name))
                continue
            else:
                out_names_done.append(out_name)

            button = tr.find_element(by=By.ID, value="match-details-modal")
            driver.execute_script("arguments[0].click();", button)

            time.sleep(3)
            # wait = WebDriverWait(driver, timeout=5)
            # wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "icon_matchReplay")))
            tbody2 = driver.find_elements(by=By.CLASS_NAME, value='icon_matchReplay')

            if len(tbody2) > 2:
                print("More than 2 players!")
                continue

            try:
                download_link = tbody2[0].get_attribute('href')
                response = requests.get(download_link, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'})
            except:
                print("first download link does not work")

                try:
                    download_link = tbody2[1].get_attribute('href')
                    response = requests.get(download_link, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'})
                except:
                    print("second download link does not work")
                    continue

            try:
                replay_zip = zipfile.ZipFile(io.BytesIO(response.content))
                replay = replay_zip.read(replay_zip.namelist()[0])
            except Exception as e:
                print(e)  # not a zip file
                continue

            # with open(PATH_OUT + out_name + '.aoe2record', 'wb') as f:
            #     f.write(replay)

            '''
            This saves the replay file and then zips it using command line. 
            Tried to dump the replay_zip above directly, but it did not work. 
            '''
            if os.path.isdir(UNZIP_FOLDER):
                shutil.rmtree(UNZIP_FOLDER)

            os.mkdir(UNZIP_FOLDER)

            full_path_unzip = UNZIP_FOLDER + out_name + '.aoe2record'
            full_path_zip = PATH_OUT + out_name + '.aoe2record.zip'
            with open(full_path_unzip, 'wb') as f:
                f.write(replay)

            os.system('zip -r ' + full_path_zip + ' ' + full_path_unzip)
            os.remove(full_path_unzip)

            # '''same code as convert_to_zip.py'''
            # binary_file_path = PATH_UNZIP + out_name + '.aoe2record'
            # zip_file_path = PATH_OUT + out_name + '.aoe2record' + '.zip'
            # with zipfile.ZipFile(zip_file_path, 'w') as f:
            #     f.write(binary_file_path)
            #
            # os.remove(PATH_UNZIP + out_name + '.aoe2record')

            print("saved a game")

            num_games += 1

    except Exception as e:
        print("general failure")
        print(e)
        continue

    if num_games > num_games_STOP:
        break

# input("Press Enter to kill")
driver.close()

time1 = time.time() - time0
print("num_games: " + str(num_games) + "  time1: " + str(time1))

