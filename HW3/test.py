# import tensorflow as tf
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
# a = tf.constant(3.0, dtype = tf.float32)
# b = tf.constant(4.0, dtype = tf.float32)
# c = a + b

# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
# z = x + y
# x = tf.placeholder(tf.float32)


# sess =tf.Session()

# # writer = tf.summary.FileWriter('summary/')
# # writer.add_graph(tf.get_default_graph())
# with tf.variable_scope("layer_1", reuse = tf.AUTO_REUSE):
# 	a1 = tf.get_variable("a", [3,4])
# with tf.control_dependencies([a1]):
#   w = a1.read_value()
# with tf.variable_scope("layer_1", reuse = tf.AUTO_REUSE):
# 	a2 = tf.get_variable("a") 

# init = tf.global_variables_initializer()
# sess.run(init)

# print(sess.run(a1))
# print(sess.run(a2))


#store data into .npy file to save time
def initTrainData(path):
	f = open(path, 'r')
	reader = csv.reader(f)
	data = list(reader)
	label = [data[i][0] for i in range(1, len(data))]
	dataSet = []
	for i in range(1, len(data)):
		pic = data[i][1]
		pic = pic.split()
		dataSet.append(pic)
	dataSet = np.array(dataSet).astype(np.float32)
	label = np.array(label).astype(np.int)

	label_onehot = np.zeros((label.shape[0], 7))
	for i in range(label.shape[0]):
		label_onehot[i][label[i]] = 1

	np.save("Data/train_x.npy", dataSet)
	np.save("Data/train_y.npy", label_onehot)

def initTestData(path):
	f = open(path, 'r')
	reader = csv.reader(f)
	data = list(reader)
	dataSet = []
	for i in range(1, len(data)):
		pic = data[i][1]
		pic = pic.split()
		dataSet.append(pic)
	dataSet = np.array(dataSet).astype(np.float32)

	np.save("Data/test_x.npy", dataSet)

#load data from .npy file
def loadDataSet(path):
	train_x = np.load(os.path.join(path,"train_x.npy"))
	train_y = np.load(os.path.join(path,"train_y.npy"))
	test_x = np.load(os.path.join(path,"test_x.npy"))
	return (train_x, train_y, test_x)

def getBatch(train_x, train_y, size):
	selected = np.random.randint(train_x.shape[0], size = size)
	batch_x = train_x[selected]
	batch_y = train_y[selected]
	return (batch_x, batch_y)

def shuffle(X, Y):
	order = np.arange(X.shape[0])
	np.random.shuffle(order)
	return (X[order], Y[order])


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv_relu(inputs, kernal_shape, bias_shape):
	weights = tf.get_variable("weights", shape = kernal_shape, initializer = tf.truncated_normal(kernal_shape, stddev = 0.1))
	bias = tf.get_variable("bias", shape = bias_shape, initializer = tf.constant(0.1, shape = kernal_shape))
	conv = tf.nn.conv2d(inputs, weights, strides = [1,1,1,1], padding = "SAME")
	return tf.nn.relu(conv + bias)

def full_connected(inputs, w_shape, b_shape):
	weights = tf.get_variable("weights", shape = w_shape, initializer = tf.truncated_normal(shape, stddev = 0.1))
	bias = tf.get_variable("bias", shape = b_shape, initializer = tf.constant(0.1, shape = b_shape))
	fc = tf.matmul(inputs, weights)+bias
	return fc


def max_pool_2x2(X):
	return tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")




def image_filter(X, keep_prob = 1.0):

	with tf.variable_scope("conv1"):
		# 48*48*1 --> 44*44*8
		conv_1 = conv_relu(X, kernal_shape = [5,5,1,8], bias_shape = [8])   
		#44*44*8 --> 22*22*8
		conv_1_pool = max_pool_2x2(conv_1)

	with tf.variable_scope("conv2"):
		#22*22*8 --> 18*18*8
		conv_2 = conv_relu(conv_1_pool, kernal_shape = [5,5,8,8], bias_shape = [8])
		#18*18*8 --> 9*9*8
		conv_2_pool = max_pool_2x2(conv_2)

	with tf.variable_scope("fc1"):
		conv_2_flat = tf.reshape(conv_2_pool, [-1, 9*9*8])
		fc1 = tf.nn.relu(full_connected(conv_2_flat, w_shape = [9*9*8, 256], b_shape = [256]))
		fc1_drop = tf.nn.dropout(fc1, keep_prob)

	with tf.variable_scope("fc2"):
		fc2 = tf.nn.softmax(full_connected(fc1_drop, w_shape = [256, 10], b_shape = [10]))

	y_conv = fc2
	return y_conv



#load data
train_x, train_y, test_x = loadDataSet("Data")
train_x = train_x.reshape((-1, 48, 48))
test_x = test_x.reshape((-1, 48, 48))

#normalize 
train_x = train_x / 255
test_x = test_x / 255








# print("show one example")
# plt.figure()
# plt.imshow(train_x[0], cmap = "gray")
# plt.show()





