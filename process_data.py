# Preprocess Sensor Data

import os
import csv
import numpy as np
from scipy import signal
import pywt

# fft
def fft_data(time, x, y, z, feats):
	for axis in [x, y, z]:
		class_count = -1

		for this_class in axis:
			class_count += 1
			data_point_count = -1

			for data_point in this_class:
				data_point_count += 1

				for sensor in data_point:

					transformed = np.fft.fft(sensor)
					max_fft = 200
					if len(transformed) > max_fft:
						transformed = transformed[0:max_fft]

					sort_transformed = sorted(abs(transformed), reverse=True)
					for i in range(5):
						feats[class_count][data_point_count].append(sort_transformed[i])
					feats[class_count][data_point_count].append(np.std(transformed))
	return feats

# psd
def psd_data(time, x, y, z, feats):
	for axis in [x, y, z]:
		class_count = -1

		for this_class in axis:
			class_count += 1
			data_point_count = -1

			for data_point in this_class:
				data_point_count += 1

				for sensor in data_point:

					_, transformed = signal.welch(sensor)
					sort_transformed = sorted(abs(transformed), reverse=True)
					for i in range(5):
						feats[class_count][data_point_count].append(sort_transformed[i])
					feats[class_count][data_point_count].append(np.std(transformed))
	return feats

# wavelet
def wavelet_data(time, x, y, z, feats):
	for axis in [x, y, z]:
		class_count = -1

		for this_class in axis:
			class_count += 1
			data_point_count = -1

			for data_point in this_class:
				data_point_count += 1

				for sensor in data_point:

					cA, cD = pywt.dwt(sensor,'db2')
					sort_transformed = sorted(abs(cD), reverse=True)
					for i in range(5):
						feats[class_count][data_point_count].append(sort_transformed[i])
					feats[class_count][data_point_count].append(np.std(cD))
	return feats

# norm features
def norm_data(time, x, y, z, feats):
	holder = [[[[[] for i in range(3)] for sensor in data_point] for data_point in this_class] for this_class in x]

	axis_count = -1
	for axis in [x, y, z]:
		class_count = -1
		axis_count += 1

		for this_class in axis:
			class_count += 1
			data_point_count = -1

			for data_point in this_class:
				data_point_count += 1
				sensor_count = -1

				for sensor in data_point:
					sensor_count += 1

					holder[class_count][data_point_count][sensor_count][axis_count] = sensor

					if axis_count == 2:
						normed = np.linalg.norm(holder[class_count][data_point_count][sensor_count], axis=0)

						feats[class_count][data_point_count].append(max(abs(normed)))
						feats[class_count][data_point_count].append(np.std(normed))
	return feats

# define features
def features(time, x, y, z, use_feats=["fft", "psd", "wavelet", "norm"]):
	# must pass in pre-sized feature vector to append
	feats = [[[] for data_point in this_class] for this_class in x] # has shape [classes, datapoints]

	if "fft" in use_feats:
		feats = fft_data(time, x, y, z, feats)

	if "psd" in use_feats:
		feats = psd_data(time, x, y, z, feats)

	if "wavelet" in use_feats:
		feats = wavelet_data(time, x, y, z, feats)

	if "norm" in use_feats:
		feats = norm_data(time, x, y, z, feats)

	combined_feats = []
	labels = []
	class_count = -1
	for this_class in feats:
		class_count += 1

		for data_point in this_class:
			combined_feats.append(data_point)
			labels.append([0 for classes in feats])
			labels[-1][class_count] = 1

	return (combined_feats, labels) # feats has shape [datapoints, features]. labels has shape [datapoints, classes]

# store data
def get_and_store(up=0, lo=1, classes = ["backhand", "forehand", "overhand_serve"], sensors = ['accelerometer', 'gyroscope'], folder_entry="Tennis Data/Training Data Combined"):
	# initialize array for storing raw data and some features
	time = [[] for this_class in classes]
	x = [[] for this_class in classes]
	y = [[] for this_class in classes]
	z = [[] for this_class in classes]

	class_count = -1

	for this_class in classes:
		# loop through each class for the test subject
		class_count += 1
		sensor_count = -1

		for sensor in sensors:
			# loop through each sensor of interest in the class
			sensor_count += 1
			data_point_count = -1

			for data_point in sorted(sorted(os.walk(folder_entry + "/" + this_class + "/" + sensor))[0][2]):
				# read data from csv file of each recording instance
				data_point_count += 1

				if sensor_count == 0:
					time[class_count].append([])
					x[class_count].append([])
					y[class_count].append([])
					z[class_count].append([])

				time[class_count][data_point_count].append([])
				x[class_count][data_point_count].append([])
				y[class_count][data_point_count].append([])
				z[class_count][data_point_count].append([])

				this_time = []
				this_x = []
				this_y = []
				this_z = []

				with open(folder_entry + "/" + this_class + "/" + sensor + "/" + data_point) as csvfile:
					data_file = csv.reader(csvfile, delimiter = ",")
					for row in data_file:
						this_time.append(np.float64(row[0]))
						this_x.append(np.float64(row[1]))
						this_y.append(np.float64(row[2]))
						this_z.append(np.float64(row[3]))

				start = int(up*len(this_time))
				end = int(lo*len(this_time))
				time[class_count][data_point_count][sensor_count] = this_time[start:end]
				x[class_count][data_point_count][sensor_count] = this_x[start:end]
				y[class_count][data_point_count][sensor_count] = this_y[start:end]
				z[class_count][data_point_count][sensor_count] = this_z[start:end]

	return (time, x, y, z)