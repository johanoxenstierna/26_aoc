
import os
import io
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

# PATH_OUT = './r/4/'
PATH_OUT = '/media/johan/KINGSTON/r/'
PATHS_DONE = ['./r/3/', '/media/johan/KINGSTON/r/']
COMPUTER_CUT = [0, 0.5]

_, _, out_names_done0 = os.walk(PATHS_DONE[0]).__next__()
_, _, out_names_done1 = os.walk(PATHS_DONE[1]).__next__()
out_names_done = out_names_done0 + out_names_done1
out_names_done = [x.split('.')[0] for x in out_names_done]
# out_names_done = []
profile_ids = get_profile_ids(3000, out_names_done, COMPUTER_CUT)

# profile_id =  #9666666combi  # 6407068#12213178reimu #1832072stefan #271202vinch #2858362jordan #666976barles  #347269accm # viper196240  # hera199325

driver = webdriver.Firefox()
time0 = time.time()
num_games = 0
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
                print(e)
                continue

            with open(PATH_OUT + out_name + '.aoe2record', 'wb') as f:
                f.write(replay)

            print("saved a game")

            num_games += 1
    except Exception as e:
        print("general failure")
        print(e)
        continue


# input("Press Enter to kill")
driver.close()

time1 = time.time() - time0
print("num_games: " + str(num_games) + "  time1: " + str(time1))

