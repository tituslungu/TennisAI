# Get sensor data from smart watch, unzip, and organize

import os
import shutil
import sys

# define and create directory for storing raw watch data
watch_path = "raw_watch_data"
if not os.path.exists(watch_path):
	os.makedirs(watch_path)

# get sensor data from watch
get_watch = input("Pull data from smart watch using adb on Linux (y/n)? ")
if get_watch == "y":

	# adb commands for Linux only
	if sys.platform == "linux":
		# os.system("sudo apt-get install android-tools-adb")
		os.system("adb kill-server")
		os.system("sudo adb start-server")
		os.system("adb pull mnt/sdcard/Sessions " + watch_path)

# define and create directory to stored unzipped and organized watch data
extracted_path = "extracted_watch_data"
if not os.path.exists(extracted_path):
	os.makedirs(extracted_path)

# create data point counter for each sensor type
data_point = {}

# loop through subfolders in the raw watch data files, then loop through the csv.gz files for each watch sensor
for swing in next(os.walk(watch_path))[1]:
	for sensor in next(os.walk(watch_path + "/" + swing + "/data"))[2]:
		sensor_folder = sensor.split(".")[2] # get name of sensor
		if sensor_folder == "csv" or sensor_folder == "android_wear":
			continue # skip watch info files

		# create directory for current sensor and initialize data point counter for this sensor
		if not os.path.exists(extracted_path + "/" + sensor_folder):
			os.makedirs(extracted_path + "/" + sensor_folder)
			data_point[sensor_folder] = 0

		# copy the zipped file to the new location, then unzip
		shutil.copy2(watch_path + "/" + swing + "/data/" + sensor, extracted_path + "/" + sensor_folder + "/" + sensor_folder + "_" + str(data_point[sensor_folder]) + ".csv.gz")
		os.system("gzip -d " + extracted_path + "/" + sensor_folder + "/" + sensor_folder + "_" + str(data_point[sensor_folder]) + ".csv.gz")

		# increment data point counter for this sensor
		data_point[sensor_folder] += 1

