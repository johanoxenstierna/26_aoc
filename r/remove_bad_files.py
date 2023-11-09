
import os
import json

# PATH = './r/6_z/'
R = 10
PATH = '/media/johan/0E45-EEA5/{}_z/'.format(str(R))

with open("./r/files_to_be_removed_{}.json".format(str(R)), 'r') as f:
    files_to_be_removed = json.load(f)

_, _, all_file_names = os.walk(PATH).__next__()

removed_files = 0
for file_name_rem in files_to_be_removed:
    if file_name_rem in all_file_names:
        print("removing " + str(file_name_rem))
        os.remove(PATH + file_name_rem)
        removed_files += 1
        print(removed_files)

_, _, all_file_names = os.walk(PATH).__next__()
for file_name in files_to_be_removed:
    if file_name in all_file_names:
        raise Exception("A file name is still there after removal")

if removed_files > 0:
    files_to_be_removed = []
    with open("./r/files_to_be_removed_PLACE.json", 'w') as f:
        json.dump(files_to_be_removed, f, indent=2)

print("removed num files: " + str(removed_files))
asdf = 5