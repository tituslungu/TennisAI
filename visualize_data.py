# Visualize Sensor Data

import os
import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy import signal
import pywt
from pan_split_seq import pan_seq_neutral, pan_seq_test

# plot data
def plot_data(x, y, z, class_plot=0, sensor_plot=0, data_point_plot=0):
	x_line, = plt.plot(x[class_plot][sensor_plot][data_point_plot], 'r')
	y_line, = plt.plot(y[class_plot][sensor_plot][data_point_plot], 'b')
	z_line, = plt.plot(z[class_plot][sensor_plot][data_point_plot], 'g')
	plt.legend([x_line, y_line, z_line], ['X', 'Y', 'Z'])
	plt.show()

# fft
def fft_data(time, x, y, z):
	x_fft = [[[np.fft.fft(k)[0:199] for k in j] for j in i] for i in x]
	y_fft = [[[np.fft.fft(k)[0:199] for k in j] for j in i] for i in y]
	z_fft = [[[np.fft.fft(k)[0:199] for k in j] for j in i] for i in z]
	return (x_fft, y_fft, z_fft)

# psd
def psd_data(time, x, y, z):
	x_psd = [[[signal.welch(k)[1] for k in j] for j in i] for i in x]
	y_psd = [[[signal.welch(k)[1] for k in j] for j in i] for i in y]
	z_psd = [[[signal.welch(k)[1] for k in j] for j in i] for i in z]
	return (x_psd, y_psd, z_psd)

# wavelet
def wavelet_data(time, x, y, z):
	x_wave1 = [[[pywt.dwt(k,'db2')[0] for k in j] for j in i] for i in x]
	y_wave1 = [[[pywt.dwt(k,'db2')[0] for k in j] for j in i] for i in y]
	z_wave1 = [[[pywt.dwt(k,'db2')[0] for k in j] for j in i] for i in z]

	x_wave2 = [[[pywt.dwt(k,'db2')[1] for k in j] for j in i] for i in x]
	y_wave2 = [[[pywt.dwt(k,'db2')[1] for k in j] for j in i] for i in y]
	z_wave2 = [[[pywt.dwt(k,'db2')[1] for k in j] for j in i] for i in z]
	return (x_wave1, y_wave1, z_wave1, x_wave2, y_wave2, z_wave2)

# store data
def get_and_store(classes = ["backhand", "forehand", "overhand_serve"], sensors = ['accelerometer', 'gyroscope']):
	data_folder = "Tennis Data"
	organized_data_folder = "Training Data Combined"

	# initialize array used for visualizing data by sensors
	time = [[[] for i in range(len(sensors))] for j in range(len(classes))]
	x = [[[] for i in range(len(sensors))] for j in range(len(classes))]
	y = [[[] for i in range(len(sensors))] for j in range(len(classes))]
	z = [[[] for i in range(len(sensors))] for j in range(len(classes))]

	data_point_count = [[-1 for i in range(len(sensors))] for j in range(len(classes))]

	class_count = -1

	for this_class in classes:
		# loop through each class for the test subject
		class_count += 1
		sensor_count = -1

		for sensor in sensors:
			# loop through each sensor of interest in the class
			sensor_count += 1

			for data_point in next(os.walk(data_folder + "/" + organized_data_folder + "/" + this_class + "/" + sensor))[2]:
				# read data from csv file of each recording instance
				data_point_count[class_count][sensor_count] += 1
				time[class_count][sensor_count].append([])
				x[class_count][sensor_count].append([])
				y[class_count][sensor_count].append([])
				z[class_count][sensor_count].append([])

				this_time = []
				this_x = []
				this_y = []
				this_z = []

				with open(data_folder + "/" + organized_data_folder + "/" + this_class + "/" + sensor + "/" + data_point) as csvfile:
					data_file = csv.reader(csvfile, delimiter = ",")
					for row in data_file:
						this_time.append(np.float64(row[0]))
						this_x.append(np.float64(row[1]))
						this_y.append(np.float64(row[2]))
						this_z.append(np.float64(row[3]))

				time[class_count][sensor_count][data_point_count[class_count][sensor_count]] = this_time
				x[class_count][sensor_count][data_point_count[class_count][sensor_count]] = this_x
				y[class_count][sensor_count][data_point_count[class_count][sensor_count]] = this_y
				z[class_count][sensor_count][data_point_count[class_count][sensor_count]] = this_z

	return (time, x, y, z)

viz = "test"
test_point = [0, 1, 39]

if viz == "swing":
	### Visualize swing data
	time, x, y, z = get_and_store(classes = ["backhand", "forehand", "overhand_serve"], sensors = ['accelerometer', 'gyroscope'])
	x_fft, y_fft, z_fft = fft_data(time, x, y, z)
	x_psd, y_psd, z_psd = psd_data(time, x, y, z)
	x_wave1, y_wave1, z_wave1, x_wave2, y_wave2, z_wave2 = wavelet_data(time,x,y,z)

	plot_data(x, y, z, class_plot=test_point[0], sensor_plot=test_point[1], data_point_plot=test_point[2])
	plot_data(x_fft, y_fft, z_fft, class_plot=test_point[0], sensor_plot=test_point[1], data_point_plot=test_point[2])
	plot_data(x_psd, y_psd, z_psd, class_plot=test_point[0], sensor_plot=test_point[1], data_point_plot=test_point[2])
	plot_data(x_wave1, y_wave1, z_wave1, class_plot=test_point[0], sensor_plot=test_point[1], data_point_plot=test_point[2])
	plot_data(x_wave2, y_wave2, z_wave2, class_plot=test_point[0], sensor_plot=test_point[1], data_point_plot=test_point[2])
elif viz == "neutral":
	### Visualize neutral data
	split_time, split_x, split_y, split_z = pan_seq_neutral(1500,300)
	x_fft, y_fft, z_fft = fft_data(split_time, split_x, split_y, split_z)
	x_psd, y_psd, z_psd = psd_data(split_time, split_x, split_y, split_z)
	x_wave1, y_wave1, z_wave1, x_wave2, y_wave2, z_wave2 = wavelet_data(split_time, split_x, split_y, split_z)

	plot_data(split_x, split_y, split_z, class_plot=0, sensor_plot=test_point[2], data_point_plot=test_point[1])
	plot_data(x_fft, y_fft, z_fft, class_plot=0, sensor_plot=test_point[2], data_point_plot=test_point[1])
	plot_data(x_psd, y_psd, z_psd, class_plot=0, sensor_plot=test_point[2], data_point_plot=test_point[1])
	plot_data(x_wave1, y_wave1, z_wave1, class_plot=0, sensor_plot=test_point[2], data_point_plot=test_point[1])
	plot_data(x_wave2, y_wave2, z_wave2, class_plot=0, sensor_plot=test_point[2], data_point_plot=test_point[1])
elif viz == "test":
	pan = 1500
	stride = 200
	test_time, test_x, test_y, test_z = pan_seq_test(pan,stride, folder_entry = "Tennis Data/Tests/daniel_lau_2")
	x_fft, y_fft, z_fft = fft_data(test_time, test_x, test_y, test_z)
	x_psd, y_psd, z_psd = psd_data(test_time, test_x, test_y, test_z)
	x_wave1, y_wave1, z_wave1, x_wave2, y_wave2, z_wave2 = wavelet_data(test_time, test_x, test_y, test_z)

	plot_data(test_x, test_y, test_z, class_plot=0, sensor_plot=test_point[2], data_point_plot=test_point[1])
	plot_data(x_fft, y_fft, z_fft, class_plot=0, sensor_plot=test_point[2], data_point_plot=test_point[1])
	plot_data(x_psd, y_psd, z_psd, class_plot=0, sensor_plot=test_point[2], data_point_plot=test_point[1])
	plot_data(x_wave1, y_wave1, z_wave1, class_plot=0, sensor_plot=test_point[2], data_point_plot=test_point[1])
	plot_data(x_wave2, y_wave2, z_wave2, class_plot=0, sensor_plot=test_point[2], data_point_plot=test_point[1])
