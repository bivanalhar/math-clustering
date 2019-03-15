"""
This program is intended to classify all the mathematical problems
by using the supervised learning method. The labelling consists of 3 labels:
Calculus, Linear Algebra and Probability & Statistics
"""

import tensorflow as tf
import numpy as np
import random, collections, time
import csv, re
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

"""
Model to be implemented : Long Short-Term Memory (LSTM)
which is believed to be able to depict the continuation of the problem's story
"""

#including all words and also its corresponding vectors
wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('wordVectors.npy')

calculus_data, linalg_data, probstat_data = [], [], []

def one_hot(label): #supporting the 3 labels for now
	return [float(label == '0'), float(label == '1'), float(label == '2')]

#Phase 1 : Preprocessing the input files
with open('eng_calculus.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ",")
	for row in csv_reader:
		pair = (row[1], one_hot(row[2]))
		calculus_data.append(pair)

with open('eng_linalg.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ",")
	for row in csv_reader:
		pair = (row[1], one_hot(row[2]))
		linalg_data.append(pair)

with open('eng_probstat.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ",")
	for row in csv_reader:
		pair = (row[1], one_hot(row[2]))
		probstat_data.append(pair)

# len_calculus = len(calculus_data)
# len_linalg = len(linalg_data)
# len_probstat = len(probstat_data)

# #setting up the dataset for training, validation and testing
# train_fetch = calculus_data[:64] + linalg_data[:64] + probstat_data[:64]
# val_fetch = calculus_data[64:96] + linalg_data[64:96] + probstat_data[64:96]
# test_fetch = calculus_data[96:128] + linalg_data[96:128] + probstat_data[96:128]

# train_data, train_label = [pair[0] for pair in train_fetch], [pair[1] for pair in train_fetch]
# val_data, val_label = [pair[0] for pair in val_fetch], [pair[1] for pair in val_fetch]
# test_data, test_label = [pair[0] for pair in test_fetch], [pair[1] for pair in test_fetch]
#######################################END OF PHASE 1########################################

strip_special = re.compile("[^A-Za-z0-9 +\-*/()=]+")

def cleanSentences(string):
	string = string.lower().replace("<br />", " ")
	return re.sub(strip_special, "", string.lower())

count_50_calculus , count_50_linalg, count_50_probstat = 0, 0, 0
count_100_calculus, count_100_linalg, count_100_probstat = 0, 0, 0

for i in range(len(calculus_data)):
	if len(cleanSentences(calculus_data[i][0]).split()) <= 50:
		count_50_calculus += 1
	if len(cleanSentences(calculus_data[i][0]).split()) <= 100:
		count_100_calculus += 1

for i in range(len(linalg_data)):
	if len(cleanSentences(linalg_data[i][0]).split()) <= 50:
		count_50_linalg += 1
	if len(cleanSentences(linalg_data[i][0]).split()) <= 100:
		count_100_linalg += 1

for i in range(len(probstat_data)):
	if len(cleanSentences(probstat_data[i][0]).split()) <= 50:
		count_50_probstat += 1
	if len(cleanSentences(probstat_data[i][0]).split()) <= 100:
		count_100_probstat += 1

print("number of sentences in Calculus less than 50 is", count_50_calculus)
print("number of sentences in Calculus less than 100 is", count_100_calculus)
print("\n")

print("number of sentences in Linear Algebra less than 50 is", count_50_linalg)
print("number of sentences in Linear Algebra less than 100 is", count_100_linalg)
print("\n")

print("number of sentences in ProbStat less than 50 is", count_50_probstat)
print("number of sentences in ProbStat less than 100 is", count_100_probstat)
print("\n")

# train_data_num, val_data_num, test_data_num = [], [], []

# for i in range(len(train_data)):
# 	numbered_sentence = [0 for i in range(100)]
# 	sentence = cleanSentences(train_data[i])
# 	sentence = sentence.split()
# 	index = 0

# 	for word in sentence:
# 		if index < 100:
# 			try:
# 				numbered_sentence[index] = wordsList.index(word)
# 			except ValueError:
# 				numbered_sentence[index] = 399999 #Vector for unknown words
# 		index += 1
# 	train_data_num.append(numbered_sentence)

# for i in range(len(val_data)):
# 	numbered_sentence = [0 for i in range(100)]
# 	sentence = cleanSentences(val_data[i])
# 	sentence = sentence.split()
# 	index = 0

# 	for word in sentence:
# 		if index < 100:
# 			try:
# 				numbered_sentence[index] = wordsList.index(word)
# 			except ValueError:
# 				numbered_sentence[index] = 399999 #Vector for unknown words
# 		index += 1
# 	val_data_num.append(numbered_sentence)

# for i in range(len(test_data)):
# 	numbered_sentence = [0 for i in range(100)]
# 	sentence = cleanSentences(train_data[i])
# 	sentence = sentence.split()
# 	index = 0

# 	for word in sentence:
# 		if index < 100:
# 			try:
# 				numbered_sentence[index] = wordsList.index(word)
# 			except ValueError:
# 				numbered_sentence[index] = 399999 #Vector for unknown words
# 		index += 1
# 	test_data_num.append(numbered_sentence)

# #write into csv file for the future network training and testing
# with open('small_train_data.csv', mode='w') as csv_file:
# 	writer = csv.writer(csv_file)

# 	for i in range(len(train_data_num)):
# 		writer.writerow(train_data_num[i])

# with open('small_val_data.csv', mode='w') as csv_file:
# 	writer = csv.writer(csv_file)

# 	for i in range(len(val_data_num)):
# 		writer.writerow(val_data_num[i])

# with open('small_test_data.csv', mode='w') as csv_file:
# 	writer = csv.writer(csv_file)

# 	for i in range(len(test_data_num)):
# 		writer.writerow(test_data_num[i])

# with open('small_train_label.csv', mode='w') as csv_file:
# 	writer = csv.writer(csv_file)

# 	for i in range(len(train_label)):
# 		writer.writerow(train_label[i])

# with open('small_val_label.csv', mode='w') as csv_file:
# 	writer = csv.writer(csv_file)

# 	for i in range(len(val_label)):
# 		writer.writerow(val_label[i])

# with open('small_test_label.csv', mode='w') as csv_file:
# 	writer = csv.writer(csv_file)

# 	for i in range(len(test_label)):
# 		writer.writerow(test_label[i])

#------------------------------------------------------------------------------------------#

# #including all words and also its corresponding vectors
# wordsList = np.load('wordsList.npy').tolist()
# wordsList = [word.decode('UTF-8') for word in wordsList]
# wordVectors = np.load('wordVectors.npy')

# #Phase 2 : Setting up the network architecture
# numClasses = 3 # Linear Algebra, Calculus, Probability and Statistics
# maxSeqLength = 100 # to keep up with some lengthy problems
# batchSize = 32 #to split the dataset into batches to prevent overflowing of the data
# lstmUnits = 512 #number of units for LSTM
# numDimensions = 50 #number of nodes in the hidden layer
# learning_rate = 1e-4 #learning rate of this model
# training_epoch = 1000
# reg_param = 0.1

# #defining the input of the network
# labels = tf.placeholder(tf.float32, [None, numClasses])
# input_data = tf.placeholder(tf.int32, [None, maxSeqLength])

# #defining the part of the network that will be used for word embedding
# with tf.device("/gpu:0"):
# 	data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype = tf.float32)
# 	data = tf.nn.embedding_lookup(wordVectors, input_data)

# 	#begin defining the LSTM Cell for the network setup
# 	#lstmCellsingle = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
# 	lstmCell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstmUnits) for _ in range(4)])
# 	lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob = 0.75)
# 	value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype = tf.float32)

# 	weight = tf.get_variable("weight", shape = [lstmUnits, numClasses], initializer = tf.contrib.layers.xavier_initializer())
# 	bias = tf.get_variable("bias", shape = [numClasses], initializer = tf.contrib.layers.xavier_initializer())
# 	value = tf.transpose(value, [1, 0, 2])
# 	last = tf.gather(value, int(value.get_shape()[0]) - 1)
# 	prediction = tf.matmul(last, weight) + bias
# 	#######################################END OF PHASE 2########################################

# 	#defining the metric used for the evaluation sake
# 	correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# 	accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# 	#defining the loss and also the optimizer for the network
# 	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels)) \
# 		   + reg_param * (tf.nn.l2_loss(weight) + tf.nn.l2_loss(bias))
# 	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
# 	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# 	init_op = tf.global_variables_initializer()

# f = open("lstm_190316_600epoch.txt", 'w')
# f.write("Result of the Experiment\n\n")

# # #Phase 3 : Setting up the starting of the network evaluation (session setup)

# # First, we need to setup the import from the csv file neatly
# train_data_sess, train_label_sess = [], []
# val_data_sess, val_label_sess = [], []
# test_data_sess, test_label_sess = [], []

# with open('train_data.csv') as csv_file:
# 	csv_reader = csv.reader(csv_file, delimiter = ",")
# 	for row in csv_reader:
# 		data_input = [int(i) for i in row]
# 		train_data_sess.append(data_input)

# with open('train_label.csv') as csv_file:
# 	csv_reader = csv.reader(csv_file, delimiter = ",")
# 	for row in csv_reader:
# 		label_input = [float(i) for i in row]
# 		train_label_sess.append(label_input)

# with open('val_data.csv') as csv_file:
# 	csv_reader = csv.reader(csv_file, delimiter = ",")
# 	for row in csv_reader:
# 		data_input = [int(i) for i in row]
# 		val_data_sess.append(data_input)

# with open('val_label.csv') as csv_file:
# 	csv_reader = csv.reader(csv_file, delimiter = ",")
# 	for row in csv_reader:
# 		label_input = [float(i) for i in row]
# 		val_label_sess.append(label_input)

# with open('test_data.csv') as csv_file:
# 	csv_reader = csv.reader(csv_file, delimiter = ",")
# 	for row in csv_reader:
# 		data_input = [int(i) for i in row]
# 		test_data_sess.append(data_input)

# with open('test_label.csv') as csv_file:
# 	csv_reader = csv.reader(csv_file, delimiter = ",")
# 	for row in csv_reader:
# 		label_input = [float(i) for i in row]
# 		test_label_sess.append(label_input)

# f.write("Setup : LSTM Units = " + str(lstmUnits) + "\n\n")
# f.write("Regularizer = " + str(reg_param) + " and Learning Rate = " + str(learning_rate) + "\n\n")
# epoch_list, cost_list = [], []

# with tf.Session() as sess:
# 	sess.run(init_op)

# 	for epoch in range(training_epoch):
# 		ptr = 0
# 		total_cost = 0.
# 		no_of_batches = int(len(train_data_sess) / batchSize)
# 		no_of_batches_val = int(len(val_data_sess) / batchSize)
# 		no_of_batches_test = int(len(test_data_sess) / batchSize)

# 		for i in range(no_of_batches):
# 			batch_in, batch_out = train_data_sess[ptr:ptr+batchSize], train_label_sess[ptr:ptr+batchSize]
# 			ptr += batchSize

# 			_, cost = sess.run([optimizer, loss], feed_dict = {input_data : batch_in, labels : batch_out})
# 			total_cost += cost / no_of_batches

# 		epoch_list.append(epoch + 1)
# 		cost_list.append(total_cost)

# 		ptr = 0
# 		acc_train, acc_val, acc_test = 0.0, 0.0, 0.0
# 		for i in range(no_of_batches):
# 			batch_in, batch_out = train_data_sess[ptr:ptr+batchSize], train_label_sess[ptr:ptr+batchSize]
# 			ptr += batchSize

# 			acc_train_part = sess.run(accuracy, feed_dict = {input_data : batch_in, labels : batch_out})
# 			acc_train += acc_train_part / no_of_batches

# 		ptr = 0
# 		for i in range(no_of_batches_val):
# 			batch_in, batch_out = val_data_sess[ptr:ptr+batchSize], val_label_sess[ptr:ptr+batchSize]
# 			ptr += batchSize

# 			acc_val_part = sess.run(accuracy, feed_dict = {input_data : batch_in, labels : batch_out})
# 			acc_val += acc_val_part / no_of_batches_val 

# 		ptr = 0
# 		for i in range(no_of_batches_test):
# 			batch_in, batch_out = test_data_sess[ptr:ptr+batchSize], test_label_sess[ptr:ptr+batchSize]
# 			ptr += batchSize

# 			acc_test_part = sess.run(accuracy, feed_dict = {input_data : batch_in, labels : batch_out})
# 			acc_test += acc_test_part / no_of_batches_test

# 		print("Epoch", epoch + 1, "finished")

# 		if epoch in [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 199, 299, 399, 499, 599]:
# 			f.write("During the " + str(epoch + 1) + "-th epoch:\n")
# 			f.write("Training Accuracy = " + str(100 * acc_train) + "\n")
# 			f.write("Validation Accuracy = " + str(100 * acc_val) + "\n")
# 			f.write("Testing Accuracy = " + str(100 * acc_test) + "\n\n")

# 	print("Optimization Finished")

# 	plt.plot(epoch_list, cost_list)
# 	plt.xlabel("Epoch")
# 	plt.ylabel("Cost Function")

# 	plt.title("LSTM Training with Regularizer and Learning Rate " + str(learning_rate))

# 	plt.savefig("LSTM_Training_190316_600epoch.png")

# 	plt.clf()
