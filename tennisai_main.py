# main entry point for project

from process_data import get_and_store, features
from pan_split_seq import pan_seq_test, pan_seq_neutral
from neural_net import nn_train
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import normalize
import sklearn.metrics as met
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import csv

# function to implement chosen model
def model_implement(train_features, train_labels, validation_features, validation_labels, classes, model_type):
	if model_type == "nn":
		### Neural Network in Tensorflow
		model, loss_all, train_all, val_all = nn_train(train_features, train_labels, classes, validation_features=validation_features, validation_labels=validation_labels)
	elif model_type == "knn":
		### KNN in SKLearn
		model = KNeighborsClassifier().fit(train_features, train_labels)
	elif model_type == "dtc":
		### Decision Tree in SKLearn
		model = DecisionTreeClassifier().fit(train_features, train_labels)
	elif model_type == "gnb":
		### Gaussian Naive Bayes in SKLearn
		train_labels = [np.where(train_labels[i]==1)[0][0] for i in range(len(train_labels))]
		validation_labels = [np.where(validation_labels[i]==1)[0][0] for i in range(len(validation_labels))]
		model = GaussianNB().fit(train_features, train_labels)
	elif model_type == "svm":
		### SVM in SKLearn
		train_labels = [np.where(train_labels[i]==1)[0][0] for i in range(len(train_labels))]
		validation_labels = [np.where(validation_labels[i]==1)[0][0] for i in range(len(validation_labels))]
		model = svm.SVC(kernel='linear', gamma=0.7, C=1.0).fit(train_features, train_labels)
	elif model_type == "rfc":
		### Random Forest in SKLearn
		model = RandomForestClassifier(n_estimators=20).fit(train_features, train_labels)
	else:
		print("ERROR: Invalid model type selected.")

	# print performance metrics if no neural net (performance of neural net printed in neural_net.py)
	if not model_type == "nn":
		train_pred = model.predict(train_features)
		val_pred = model.predict(validation_features)
		train_all = met.accuracy_score(train_labels, train_pred)
		val_all = met.accuracy_score(validation_labels, val_pred)
		print('Training Accuracy {}'.format(train_all) + ', Validation Accuracy {}'.format(val_all))

	return (model, train_all, val_all)

# Start

### SELECT MODEL TO IMPLEMENT ###
# options: nn, knn, dtc, gnb, svm, rfc
model_type = "nn"

# identify classes, sensors to use, features to extract, and data location
classes = ["backhand", "forehand", "overhand_serve"] # neutral must not be specified here, but as a boolean below
use_neutral = True
sensors = ['accelerometer', 'gyroscope']
use_feats = ["fft", "psd", "wavelet", "norm"]
folder_entry = "Tennis Data/Training Data Combined"

# test on game data sequence at the end?
test_on_vid = False

# get swing and serve data
time, x, y, z = get_and_store(up=0, lo=1, classes=classes, sensors=sensors, folder_entry=folder_entry)

if use_neutral:
	# add neutral data
	classes.append("neutral")
	time_neutral, x_neutral, y_neutral, z_neutral = pan_seq_neutral(1500, 300, sensors=sensors)
	time.append(*time_neutral)
	x.append(*x_neutral)
	y.append(*y_neutral)
	z.append(*z_neutral)

# extract features from data
feats, feat_labels = features(time, x, y, z, use_feats=use_feats)
feats = np.array(feats)
feat_labels = np.array(feat_labels)

# normalize features
feats = normalize(feats)

# implement k-fold cross validation?
do_kf = True

if do_kf:
	### K Fold Cross Validation
	k = 10
	k_count = -1
	train_all = [[] for i in range(k)]
	val_all = [[] for i in range(k)]

	kf = KFold(n_splits=k, shuffle=True)
	for train_index, val_index in kf.split(feats):
		k_count += 1
		model, train_all[k_count], val_all[k_count] = model_implement(feats[train_index], feat_labels[train_index], feats[val_index], feat_labels[val_index],
																						classes, model_type)
	train_all = np.mean(train_all, axis=0)
	val_all = np.mean(val_all, axis=0)
else:
	### Split train/validation
	train_features, validation_features, train_labels, validation_labels = train_test_split(feats, feat_labels, test_size=0.2, random_state=832289)
	model, train_all, val_all = model_implement(train_features, train_labels, validation_features, validation_labels, classes, model_type)

### show performance metrics
if model_type == "nn":
	print('\nFinal Average: Training Accuracy {}'.format(train_all[-1]) + ', Validation Accuracy {}'.format(val_all[-1]))

	### Plot Results
	acc_plot = plt
	acc_plot.title('Accuracy')
	acc_plot.plot(train_all, 'r', label='Training Accuracy')
	acc_plot.plot(val_all, 'b', label='Validation Accuracy')
	acc_plot.legend(loc=4)
	acc_plot.xlabel('Epochs')
	acc_plot.ylabel('Accuracy')
	plt.tight_layout()
	plt.show()
else:
	print('\nFinal Average: Training Accuracy {}'.format(train_all) + ', Validation Accuracy {}'.format(val_all))

### test on game data sequence
if test_on_vid:
	# shuffle training data
	feats, _, feat_labels, _ = train_test_split(feats, feat_labels, test_size=0.0, random_state=832289)

	# sliding window on test sequence
	pan = 1000
	stride = 200
	test_time, test_x, test_y, test_z = pan_seq_test(pan,stride, sensors=sensors, folder_entry = "Tennis Data/Tests/daniel_lau_2")

	# extract and normalize features for test sequence
	test_feats, _ = features(test_time, test_x, test_y, test_z, use_feats=use_feats)
	test_feats = np.array(test_feats)
	test_feats = normalize(test_feats)

	# train model on all data and predict on test sequence
	if not model_type == "nn":
		if model_type == "gnb" or model_type == "svm":
			feat_labels = [np.where(feat_labels[i]==1)[0][0] for i in range(len(feat_labels))]
		model.fit(feats, feat_labels)
		predict_test = model.predict(test_feats)

		if not np.array(predict_test)[0].shape == ():
			preds = predict_test
			predict_test = []

			for i in range(len(preds)):
				if 1 in preds[i]:
					predict_test.append(classes[np.where(np.array(preds)[i]==1)[0][0]])
				else:
					predict_test.append('null')

		else:
			predict_test = [classes[prediction] for prediction in predict_test]
	else:
		predict_test = nn_train(feats, feat_labels, classes, test_feats=test_feats, mode="test")
		predict_test = [classes[prediction] for prediction in predict_test]

	# save predicions to csv file
	with open('video_predictions.csv', 'w', newline='') as csvfile:
		testwriter = csv.writer(csvfile, delimiter=',')
		predict_time = 0
		for prediction in predict_test:
			testwriter.writerow([predict_time, prediction])
			predict_time += stride
