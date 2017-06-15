# Pan test sequence of tennis game to run classifier and identify swings/states
# This function assumes that there is only one continuous test sequence at the specified location, ie under each sensor there should only be one file containing
# the sensor readings from that test
# 
# the "pan_seq_test" and "pan_seq_neutral" functions below are in fact identical, only difference is some pre-set parameters
# 
# FIRST TIMESTAMP ON FIRST SENSOR IS TAKEN AS STARTING POINT AND ALL BUCKETS ARE COMPUTED RESPECTIVE TO IT (BASED ON PAN AND STRIDE) FOR EACH SENSOR IN THAT CLASS AND DATAPOINT
# 
# Sampling Frequencies, pan, and stride should still be compatible, ie the pan should be much smaller than the length of a data sequence of the sensor with
# the smallest sampling frequency

import os
import csv
import numpy as np
from process_data import get_and_store

# pan and stride should both be in milliseconds
def pan_seq_test(pan, stride, sensors=['accelerometer', 'gyroscope'], classes = ["Organized Data"], folder_entry = "Tennis Data/Tests/daniel_lau_1"):

	# retrieve and sort data. Each list output is in the form [class, datapoint, sensor]
	time, x, y, z = get_and_store(classes=classes, sensors=sensors, folder_entry=folder_entry)

	# test
	# time = [[[[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]],[[11,12,13,14,15,16,17,18,19,20],[11,12,13,14,15,16,17,18,19,20]]]]
	# x = [[[[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]],[[21,22,23,24,25,26,27,28,29,30],[31,32,33,34,35,36,37,38,39,40]]]]
	# y = [[[[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]],[[21,22,23,24,25,26,27,28,29,30],[31,32,33,34,35,36,37,38,39,40]]]]
	# z = [[[[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]],[[21,22,23,24,25,26,27,28,29,30],[31,32,33,34,35,36,37,38,39,40]]]]

	# initiliaze lists to store split data
	split_time = [[] for this_class in classes]
	split_x = [[] for this_class in classes]
	split_y = [[] for this_class in classes]
	split_z = [[] for this_class in classes]

	class_count = -1

	# iterate through each level of the data
	for this_class in time:
		class_count += 1
		data_point_count = -1
		new_data_master = -1

		for data_point in this_class:
			data_point_count += 1
			sensor_count = -1

			for sensor in data_point:
				sensor_count += 1
				new_data_count = new_data_master

				# use the first sensor to determine the number of new data points that will be created
				if sensor_count == 0:
					passes = int(len(time[class_count][data_point_count][sensor_count])/stride)

				# loop for each new data point being created
				for increment in range(passes):
					new_data_count += 1

					if sensor_count == 0:
						# use the first time stamp of the first sensor as your temporal start point
						start_time = time[class_count][data_point_count][sensor_count][0]

						# for the first sensor, a new data point is added to the list. That data point is revisited when looping through subsequent sensors
						split_time[class_count].append([])
						split_x[class_count].append([])
						split_y[class_count].append([])
						split_z[class_count].append([])

					# add a new entry to the data point for this new sensor
					split_time[class_count][new_data_count].append([])
					split_x[class_count][new_data_count].append([])
					split_y[class_count][new_data_count].append([])
					split_z[class_count][new_data_count].append([])

					# find the limits of the data that need to be saved for this new data point
					start = start_time + increment*stride
					end = start_time + increment*stride + pan

					start = [abs(i - start) for i in time[class_count][data_point_count][sensor_count]]
					start = start.index(min(start))

					end = [abs(i - end) for i in time[class_count][data_point_count][sensor_count]]
					end = end.index(min(end))

					# retrieve the data from the original data vector and store as this new data point
					split_time[class_count][new_data_count][sensor_count] = time[class_count][data_point_count][sensor_count][start:end]
					split_x[class_count][new_data_count][sensor_count] = x[class_count][data_point_count][sensor_count][start:end]
					split_y[class_count][new_data_count][sensor_count] = y[class_count][data_point_count][sensor_count][start:end]
					split_z[class_count][new_data_count][sensor_count] = z[class_count][data_point_count][sensor_count][start:end]

			# update the master data counter, which ensures that new data points are created and appended for any number of original data points,
			# while still allowing each new data point to be revisited for every sensor
			new_data_master = new_data_count

	# Each list output is in the form [class, datapoint, sensor]
	return (split_time, split_x, split_y, split_z)

def pan_seq_neutral(pan, stride, sensors=['accelerometer', 'gyroscope'], classes = ["neutral"], folder_entry = "Tennis Data/Training Data Combined"):
	
	# call pan_seq_test with a different path for splitting the neural state
	split_time, split_x, split_y, split_z = pan_seq_test(pan, stride, sensors=sensors, classes = classes, folder_entry = folder_entry)

	# Each list output is in the form [class, datapoint, sensor]
	return (split_time, split_x, split_y, split_z)