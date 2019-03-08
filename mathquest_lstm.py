"""
This program is intended to classify all the mathematical problems
by using the supervised learning method. The labelling consists of 3 labels:
Calculus, Linear Algebra and Probability & Statistics
"""

import tensorflow as tf
import numpy as np
import random, collections, time
import csv

"""
Model to be implemented : Long Short-Term Memory (LSTM)
which is believed to be able to depict the continuation of the problem's story
"""

#including all words and also its corresponding vectors
wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('wordVectors.npy')

numClasses = 3 # Linear Algebra, Calculus, Probability and Statistics
maxSeqLength = 200 # to keep up with some lengthy problems
batchSize = 32
lstmUnits = 64
numDimensions = 10

#start to create the graph for the evaluation
tf.reset_default_graph()

#defining the input of the network
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

#defining the part of the network that will be used for word embedding
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype = tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)

#begin defining the LSTM Cell for the network setup
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob = 0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype = tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape = [numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = tf.matmul(last, weight) + bias

#defining the metric used for the evaluation sake
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

#defining the loss and also the optimizer for the network
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)
