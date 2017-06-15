# Tensorflow Neural Net

import tensorflow as tf
import numpy as np

# function to create deep neural network
def deep_net(nodes_per_layer, act_per_layer, keep_prob, nnet, method, reg=0.0):
	regularizer = tf.contrib.layers.l2_regularizer(reg)
	if method == 1:
		### Use contrib library to create a fully connected layer
		for i in range(len(nodes_per_layer)):
			nnet = tf.contrib.layers.fully_connected(nnet, nodes_per_layer[i], activation_fn=None, weights_regularizer=regularizer)
			if act_per_layer[i] == "sigmoid":
				nnet = tf.sigmoid(nnet)
			elif act_per_layer[i] == "softmax":
				nnet = tf.nn.softmax(nnet)
			elif act_per_layer[i] == "relu":
				nnet = tf.nn.relu(nnet)
			elif act_per_layer[i] == None:
				continue
			else:
				print("WARNING: Invalid option passed for activation function of layer number " + str(i) + ", continuing with linear activation.")
			nnet = tf.nn.dropout(nnet, keep_prob=keep_prob[i])
	elif method == 2:
		# Use layers library to create a dense layer
		for i in range(len(nodes_per_layer)):
			nnet = tf.layers.dense(nnet, nodes_per_layer[i], activation=None, weights_regularizer=regularizer)
			if act_per_layer[i] == "sigmoid":
				nnet = tf.sigmoid(nnet)
			elif act_per_layer[i] == "softmax":
				nnet = tf.nn.softmax(nnet)
			elif act_per_layer[i] == "relu":
				nnet = tf.nn.relu(nnet)
			elif act_per_layer[i] == None:
				continue
			else:
				print("WARNING: Invalid option passed for activation function of layer number " + str(i) + ", continuing with linear activation.")
			nnet = tf.nn.dropout(nnet, keep_prob=keep_prob[i])
	elif method == 3:
		for i in range(len(nodes_per_layer)):
			# Manual NN Implementation
			weights = tf.Variable(tf.truncated_normal([nnet.get_shape().as_list()[1], nodes_per_layer[i]], stddev=0.35, mean=0.0), name="weights")
			biases = tf.Variable(tf.zeros([nodes_per_layer[i]]), name="biases")
			nnet = tf.matmul(nnet,weights) + biases
			if act_per_layer[i] == "sigmoid":
				nnet = tf.sigmoid(nnet)
			elif act_per_layer[i] == "softmax":
				nnet = tf.nn.softmax(nnet)
			elif act_per_layer[i] == "relu":
				nnet = tf.nn.relu(nnet)
			elif act_per_layer[i] == None:
				continue
			else:
				print("WARNING: Invalid option passed for activation function of layer number " + str(i) + ", continuing with linear activation.")
			nnet = tf.nn.dropout(nnet, keep_prob=keep_prob[i])
	else:
		print("WARNING: Invalid option passed for implementation method ID, continuing with method 1.")
		nnet = deep_net(nodes_per_layer, act_per_layer, nnet, 1)

	return nnet

# Main NN function
def nn_train(train_features, train_labels, classes, validation_features=None, validation_labels=None, test_feats=None, mode="train"):
	tf.reset_default_graph()

	lr = 0.001 # learning rate
	epochs = 400

	inputs = tf.placeholder(tf.float32, (None, len(train_features[0])))
	labels = tf.placeholder(tf.float32, (None, len(classes)))

	### Specify size of NN
	# nodes = [30, 50, 75, 50, 30, 20, 10, 5, len(classes)] # overkill, and overfits
	nodes = [50, 30, 20, 10, len(classes)] # 99% train, 96% validate for 3 classes
	# nodes = [75, 50, 30, 20, 10, 5, len(classes)] # 99% train, 96-97% validate for 3 classes
	# nodes = [90, 75, 50, 30, 20, 10, 5, len(classes)] # ???

	### Specify architecture of NN
	activs = [*["relu"]*(len(nodes)-1), "softmax"]
	keep_probs = [1]*len(nodes)
	nnet = deep_net(nodes, activs, keep_probs, inputs, 3)

	### Cost function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nnet, labels=labels))

	### Regularization methods
	# regularizer = tf.nn.l2_loss(weights)
	# cost = tf.reduce_mean(cost + 0.01*regularizer)

	# regularizer = tf.contrib.layers.l2_regularizer(0.01)
	# reg_penalty = tf.contrib.layers.apply_regularization(regularizer)
	# cost = tf.reduce_mean(cost + reg_penalty)

	### Optimizer options
	# optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
	# optimizer = tf.train.AdagradOptimizer(lr).minimize(cost)
	optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
	# optimizer = tf.train.MomentumOptimizer(lr, 0.5).minimize(cost)

	### Evaluate perfomance
	is_correct_prediction = tf.equal(tf.argmax(nnet, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

	### Get predicted label
	eval_test = tf.argmax(nnet, 1)

	loss_all = []
	train_all = []
	val_all = []

	### Specify batch size
	batch_size = len(train_features)
	batch_counter = 0

	print('')

	if mode == "train":
		### Training mode
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for epoch in range(epochs):

				for batch in range(int(len(train_features)/batch_size)):
					batch_counter += 1

					batch_features = train_features[batch*batch_size:(batch+1)*batch_size]
					batch_labels = train_labels[batch*batch_size:(batch+1)*batch_size]

					_, loss_val = sess.run([optimizer, cost], feed_dict = {inputs: batch_features, labels: batch_labels}) # train net on training examples

					train_acc = sess.run(accuracy, feed_dict = {inputs: train_features, labels: train_labels})
					val_acc = sess.run(accuracy, feed_dict = {inputs: validation_features, labels: validation_labels})

					loss_all.append(loss_val)
					train_all.append(train_acc)
					val_all.append(val_acc)

					if not batch_counter % 50:
						print('Epoch {}'.format(epoch+1) + ', Loss {}'.format(loss_all[-1]) + ', Training Accuracy {}'.format(train_all[-1]) + ', Validation Accuracy {}'.format(val_all[-1]))

		return (nnet, loss_all, train_all, val_all)

	else:
		### Testing mode
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for epoch in range(epochs):

				for batch in range(int(len(train_features)/batch_size)):
					batch_counter += 1

					batch_features = train_features[batch*batch_size:(batch+1)*batch_size]
					batch_labels = train_labels[batch*batch_size:(batch+1)*batch_size]

					_, loss_val = sess.run([optimizer, cost], feed_dict = {inputs: batch_features, labels: batch_labels}) # train net on training examples

			predict_test = sess.run(eval_test, feed_dict={inputs: test_feats})

		return predict_test