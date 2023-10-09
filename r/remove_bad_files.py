
import os
import json

PATH = './r/3/'
with open("./r/files_to_be_removed.json", 'r') as f:
    files_to_be_removed = json.load(f)

_, _, all_file_names = os.walk(PATH).__next__()

for file_name in files_to_be_removed:
    if file_name in all_file_names:
        print("removing " + str(file_name))
        os.remove(PATH + file_name)

_, _, all_file_names = os.walk(PATH).__next__()
for file_name in files_to_be_removed:
    if file_name in all_file_names:
        raise Exception("A file name is still there after removal")

files_to_be_removed = []
with open("./r/files_to_be_removed.json", 'w') as f:
    json.dump(files_to_be_removed, f, indent=2)


asdf = 5