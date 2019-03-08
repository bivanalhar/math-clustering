"""
This program is intended to classify all the mathematical problems
by using the supervised learning method. The labelling consists of 3 labels:
Calculus, Linear Algebra and Probability & Statistics
"""

# import tensorflow as tf
import numpy as np
import random, collections, time
import csv, re

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

len_calculus = len(calculus_data)
len_linalg = len(linalg_data)
len_probstat = len(probstat_data)

#setting up the dataset for training, validation and testing
train_fetch = calculus_data[:int(0.7*len_calculus)] + linalg_data[:int(0.7*len_linalg)] \
	+ probstat_data[:int(0.7*len_probstat)]
val_fetch = calculus_data[int(0.7*len_calculus):int(0.85*len_calculus)] \
	+ linalg_data[int(0.7*len_linalg):int(0.85*len_linalg)] \
	+ probstat_data[int(0.7*len_probstat):int(0.85*len_probstat)]
test_fetch = calculus_data[int(0.85*len_calculus):] + linalg_data[int(0.85*len_linalg):] \
	+ probstat_data[int(0.85*len_probstat):]

train_data, train_label = [pair[0] for pair in train_fetch], [pair[1] for pair in train_fetch]
val_data, val_label = [pair[0] for pair in val_fetch], [pair[1] for pair in val_fetch]
test_data, test_label = [pair[0] for pair in test_fetch], [pair[1] for pair in test_fetch] 
#######################################END OF PHASE 1########################################

#Testing the length of string to determine the maximum sequential length
# num_words = []
# for sentence in train_data + val_data + test_data:
# 	num_words.append(len(sentence.split()))
# print(max(num_words))

strip_special = re.compile("[^A-Za-z0-9 +\-*/()]+")

def cleanSentences(string):
	string = string.lower().replace("<br />", " ")
	return re.sub(strip_special, "", string.lower())

train_data_num, val_data_num, test_data_num = [], [], []

for i in range(len(train_data)):
	numbered_sentence = [0 for i in range(100)]
	sentence = cleanSentences(train_data[i])
	sentence = sentence.split()
	index = 0

	for word in sentence:
		if index < 100:
			try:
				numbered_sentence[index] = wordsList.index(word)
			except ValueError:
				numbered_sentence[index] = 399999 #Vector for unknown words
		index += 1
	train_data_num.append((numbered_sentence, train_label[i]))

for i in range(len(val_data)):
	numbered_sentence = [0 for i in range(100)]
	sentence = cleanSentences(val_data[i])
	sentence = sentence.split()
	index = 0

	for word in sentence:
		if index < 100:
			try:
				numbered_sentence[index] = wordsList.index(word)
			except ValueError:
				numbered_sentence[index] = 399999 #Vector for unknown words
		index += 1
	val_data_num.append((numbered_sentence, val_label[i]))

for i in range(len(test_data)):
	numbered_sentence = [0 for i in range(100)]
	sentence = cleanSentences(train_data[i])
	sentence = sentence.split()
	index = 0

	for word in sentence:
		if index < 100:
			try:
				numbered_sentence[index] = wordsList.index(word)
			except ValueError:
				numbered_sentence[index] = 399999 #Vector for unknown words
		index += 1
	test_data_num.append((numbered_sentence, test_label[i]))

#write into csv file for the future network training and testing
with open('train_data.csv', mode='w') as csv_file:
	fieldnames = ['vector_word', 'one-hot label']
	writer = csv.DictWriter(csv_file, fieldnames = fieldnames)

	writer.writeheader()
	for i in range(len(train_data_num)):
		writer.writerow({'vector_word' : train_data_num[i][0], 'one-hot label' : train_data_num[i][1]})

with open('val_data.csv', mode='w') as csv_file:
	fieldnames = ['vector_word', 'one-hot label']
	writer = csv.DictWriter(csv_file, fieldnames = fieldnames)

	writer.writeheader()
	for i in range(len(val_data_num)):
		writer.writerow({'vector_word' : val_data_num[i][0], 'one-hot label' : val_data_num[i][1]})

with open('test_data.csv', mode='w') as csv_file:
	fieldnames = ['vector_word', 'one-hot label']
	writer = csv.DictWriter(csv_file, fieldnames = fieldnames)

	writer.writeheader()
	for i in range(len(test_data_num)):
		writer.writerow({'vector_word' : test_data_num[i][0], 'one-hot label' : test_data_num[i][1]})

# #including all words and also its corresponding vectors
# wordsList = np.load('wordsList.npy').tolist()
# wordsList = [word.decode('UTF-8') for word in wordsList]
# wordVectors = np.load('wordVectors.npy')

# #Phase 2 : Setting up the network architecture
# numClasses = 3 # Linear Algebra, Calculus, Probability and Statistics
# maxSeqLength = 100 # to keep up with some lengthy problems
# batchSize = 32 #to split the dataset into batches to prevent overflowing of the data
# lstmUnits = 64 #number of units for LSTM
# numDimensions = 10 #number of nodes in the hidden layer
# training_epoch = 50

# #defining the input of the network
# labels = tf.placeholder(tf.float32, [batchSize, numClasses])
# input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

# #defining the part of the network that will be used for word embedding
# data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype = tf.float32)
# data = tf.nn.embedding_lookup(wordVectors, input_data)

# #begin defining the LSTM Cell for the network setup
# lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
# lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob = 0.75)
# value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype = tf.float32)

# weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
# bias = tf.Variable(tf.constant(0.1, shape = [numClasses]))
# value = tf.transpose(value, [1, 0, 2])
# last = tf.gather(value, int(value.get_shape()[0]) - 1)
# prediction = tf.matmul(last, weight) + bias
# #######################################END OF PHASE 2########################################


# #defining the metric used for the evaluation sake
# correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# #defining the loss and also the optimizer for the network
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
# optimizer = tf.train.AdamOptimizer().minimize(loss)

# init_op = tf.global_variables_initializer()

# # #Phase 3 : Setting up the starting of the network evaluation (session setup)

# # with tf.Session() as sess:
# # 	sess.run(init_op)

# # 	for epoch in range(training_epoch):
# # 		ptr = 0
# # 		no_of_batches = int(len(train_data) / batchSize)

# # 		for i in range(no_of_batches):
# # 			batch_in, batch_out = 