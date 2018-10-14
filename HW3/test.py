# import tensorflow as tf
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import tensorflow as tf
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
	weights = tf.get_variable("weights",  initializer = tf.truncated_normal(kernal_shape, stddev = 0.1))
	tf.summary.histogram('weights',weights)
	bias = tf.get_variable("bias", initializer = tf.constant(0.1, shape = bias_shape))
	tf.summary.histogram("bias",bias)
	conv = tf.nn.conv2d(inputs, weights, strides = [1,1,1,1], padding = "SAME")
	return tf.nn.relu(conv + bias)

def full_connected(inputs, w_shape, b_shape):
	weights = tf.get_variable("weights", initializer = tf.truncated_normal(w_shape, stddev = 0.1))
	bias = tf.get_variable("bias", initializer = tf.constant(0.1, shape = b_shape))
	tf.summary.histogram("weights",weights)
	tf.summary.histogram("bias", bias)
	fc = tf.matmul(inputs, weights)+bias
	return fc


def max_pool(X, window_size, strides):
	return tf.nn.max_pool(X, ksize = [1,window_size,window_size,1], strides = [1,strides,strides,1], padding = "VALID")




def image_filter(X, keep_prob):

	with tf.variable_scope("conv1"):

		conv_1 = conv_relu(X, kernal_shape = [1,1,1,32], bias_shape = [32])   
		#conv_1_pool = max_pool_2x2(conv_1)  

	with tf.variable_scope("conv2"):

		conv_2 = conv_relu(conv_1, kernal_shape = [5,5,32,32], bias_shape = [32])
		conv_2_pool = max_pool(conv_2, 3, 2)  #23

	with tf.variable_scope("conv3"):
	
		conv_3 = conv_relu(conv_2_pool, kernal_shape = [3,3,32,32], bias_shape = [32])	
		conv_3_pool = max_pool(conv_3, 3, 2)  #11


	with tf.variable_scope("conv4"):
	
		conv_4 = conv_relu(conv_3_pool, kernal_shape = [5,5,32,64], bias_shape = [64])	
		conv_4_pool = max_pool(conv_4, 3, 2)  #5

	with tf.variable_scope("fc1"):
		conv_4_flat = tf.reshape(conv_4_pool, [-1, 5*5*64])
		fc1 = tf.nn.relu(full_connected(conv_4_flat, w_shape = [5*5*64,1024], b_shape = [1024]))
		fc1_drop = tf.nn.dropout(fc1, keep_prob)

	with tf.variable_scope("fc2"):
		fc2 = tf.nn.relu(full_connected(fc1_drop, w_shape = [1024, 128], b_shape = [128]))
		fc2_drop = tf.nn.dropout(fc2, keep_prob)

	with tf.variable_scope("output"):
		y_conv = tf.nn.softmax(full_connected(fc2_drop, w_shape = [128, 7], b_shape = [7]))

	return y_conv



#load data
train_x, train_y, test_x = loadDataSet("Data")
train_x, train_y = shuffle(train_x, train_y)
train_x = train_x.reshape((-1, 48, 48))
test_x = test_x.reshape((-1, 48, 48))

print(train_x.shape)
#normalize 
train_x = train_x / 255.0
test_x = test_x / 255.0

# train_x = train_x[0 : int(train_x.shape[0]*0.8)]
# valid_x = train_x[int(train_x.shape[0]):]

x = tf.placeholder("float", shape = [None,48,48])
y = tf.placeholder("float", shape = [None, 7])
keep_prob = tf.placeholder("float")

x_image = tf.reshape(x, [-1, 48, 48, 1])
y_conv = image_filter(x_image, keep_prob = keep_prob)

cross_entropy = -tf.reduce_mean(y*tf.log(y_conv))
tf.summary.scalar("loss", cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
y_pred = tf.argmax(y_conv,1)
prediction_comp = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction_comp, "float"))

sess = tf.Session()

writer = tf.summary.FileWriter("log/", sess.graph)
sess.run(tf.initialize_all_variables())

valid_x = train_x[27200:28709]
valid_y = train_y[27200:28709]

for epoch in range(50):
	for i in range(850):
		batch_x = train_x[i*32:(i+1)*32]
		batch_y = train_y[i*32:(i+1)*32]
		_, acc = sess.run((train_step, accuracy), feed_dict = {x:batch_x, y:batch_y, keep_prob: 0.5})
	
	acc, rs =  sess.run((accuracy,  tf.summary.merge_all()), feed_dict = {x:valid_x, y:valid_y, keep_prob:1.0})
	writer.add_summary(rs,epoch)
	print(acc)






# print("show one example")
# plt.figure()
# plt.imshow(train_x[0], cmap = "gray")
# plt.show()





