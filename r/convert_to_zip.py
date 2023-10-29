
import os
import io
import zipfile

PATH_IN = './r/4/'
PATH_OUT = './r/4_z/'

_, _, file_names = os.walk(PATH_IN).__next__()

for i, file_name in enumerate(file_names):

	full_path_unzip = PATH_IN + file_name
	full_path_zip = PATH_OUT + file_name + '.zip'

	os.system('zip -r ' + full_path_zip + ' ' + full_path_unzip)

	print(str(i) + " out of " + str(len(file_names)) + " done")


	# PEND DEL
	# with open(PATH_IN + file_name, 'rb') as f:
	# 	m_b = f.open(f)

	# replay_zip = zipfile.ZipFile(io.BytesIO(response.content))

	# z = zipfile.ZipFile(PATH_IN + file_name, 'w')
	# z.writestr(PATH_OUT + file_name, bytes)
	# z.close()

	# binary_file_path = PATH_IN + file_name
	# zip_file_path = PATH_OUT + file_name + '.zip'
	# with zipfile.ZipFile(zip_file_path, 'w') as f:
	# 	f.write(binary_file_path)

	# aa = 5
