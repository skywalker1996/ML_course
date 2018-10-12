import numpy as np
import re
import math
import argparse
import os

learningRate = 0.1
batch_size = 32
epoch_num = 200

def getData(path):
	data = []
	f = open(path , "r")
	line = f.readline()
	line = f.readline()
	m = re.compile(r'\d+')
	while(line):
		elements = m.findall(line)
		elements = [float(num) for num in elements]
		data.append(elements)
		line = f.readline()
	f.close()
	return data

def getBatch(train_X, train_Y, size):
	select_n = np.random.randint(len(train_X), size = size)
	batch_X = train_X[select_n]
	batch_Y = train_Y[select_n]
	return (batch_X, batch_Y)

def sigmoid(z):
	y = 1/(1.0 + np.exp(-z))
	return np.clip(y, 1e-8, 1-(1e-8))
	
def shuffle(X, Y):
	order = np.arange(X.shape[0])
	np.random.shuffle(order)
	return (X[order], Y[order])

def normalize(train_x, test_x):
	data = np.concatenate((train_x, test_x))
	mu = np.sum(data, axis = 0)/data.shape[0]
	sigma = np.std(data, axis = 0)
	mu = np.tile(mu, (data.shape[0], 1))
	sigma = np.tile(sigma, (data.shape[0], 1))
	data = (data - mu) / sigma

	return data[:train_x.shape[0]], data[train_x.shape[0]:]

def valid(X, Y, w, b):
	data_size = X.shape[0]
	z = np.dot(X, np.transpose(w)) + b
	y_ = sigmoid(z)
	y_ = np.around(y_)
	result = (y_ == Y)

	print("the accuracy is " + str(result.sum()/data_size) if(data_size!=0) else 0)

def split_train_valid(train_all, label_all, percentage):
	train_all, label_all = shuffle(train_all, label_all)
	dataSize = train_all.shape[0]
	line = int(np.floor(dataSize * percentage))

	train_X = train_all[0 : line]
	valid_X = train_all[line : ]
	train_Y = label_all[0 : line]
	valid_Y = label_all[line : ]

	return train_X, train_Y, valid_X, valid_Y



# w = np.zeros(106)
# b = np.zeros(1)
# valid(train_X, train_Y, w, b)

def train_func(train_X, train_Y, opts ):


	train_X, train_Y, valid_X, valid_Y = split_train_valid(train_X, train_Y, 0.9)
	w = np.zeros(106)
	y = np.zeros(25)
	b = np.zeros(1)
	z = np.zeros(25)
	cost = 0
	a_wgrad = 0
	a_bgrad = 0
	save_iter_num = 10
	total_loss = 0.0
	step_num = int(np.floor(train_X.shape[0]/batch_size))


	for epoch in range(epoch_num):

		if(epoch%save_iter_num == 0):
			print("=======save the parameters at epoch %d=======" % epoch)
			if( not os.path.exists(opts.save_dir)):
				os.mkdir(opts.save_dir)
			else:
				np.savetxt(os.path.join(opts.save_dir, "w"), w)
				np.savetxt(os.path.join(opts.save_dir, "b"), b)
				avg_loss = total_loss / (save_iter_num * train_X.shape[0])
				print("===average loss is %f===" % avg_loss)
				total_loss = 0.0
				valid(valid_X, valid_Y, w, b)


		for idx in range(step_num):
			#batch_X, batch_Y = getBatch(train_X, train_Y, size = batch_size) 
			batch_X = train_X[idx*batch_size : (idx+1)*batch_size]
			batch_Y = train_Y[idx*batch_size : (idx+1)*batch_size] 
			z = np.dot(batch_X, w.T) + b
			#print("the w is " + str(w))
			#print("the z is " + str(z))
			y = sigmoid(z)
			#print("the y is " + str(y))
			cross_entropy = -1*(np.dot(batch_Y, np.log(y)) + np.dot((1-batch_Y), np.log(1-y)))
			total_loss += cross_entropy

			w_grad = np.mean(-1 * batch_X * (batch_Y - y).reshape(batch_size,1), axis = 0)
			b_grad = np.mean(-1 * (batch_Y - y))
			#print("the w_grad is " + str(w_grad))
			#print("the b_grad is " + str(b_grad))
			a_wgrad += w_grad**2
			a_bgrad += b_grad**2
			ada_w = np.sqrt(a_wgrad)
			ada_b = np.sqrt(a_bgrad)
			w = w - learningRate * w_grad
			b = b - learningRate * b_grad
			#print("the cost is " + str(cost))


def infer_func(test_X, opts):
	print("=====load the parameters=====")
	w = np.loadtxt(os.path.join(opts.save_dir, "w"))
	b = np.loadtxt(os.path.join(opts.save_dir, "b"))
	z = np.dot(test_X, np.transpose(w)) + b
	y = sigmoid(z)
	y_ = np.around(y)

	print("=====save the result into ans.csv=====")
	if(not os.path.exists(opts.output_dir)):
		os.mkdir(opts.output_dir)
	else:
		output_path = os.path.join(opts.output_dir, "ans.csv")
		with open(output_path, "w+") as f:
			f.write("id,label\n")
			count = 1
			for value in y_:
				f.write("%d,%d\n" % (count, value))
				count += 1
	return 

def main(opts):
	train_X = getData(opts.train_data_path)
	train_Y = getData(opts.train_label_path)
	test_X = getData(opts.test_data_path)
	train_X = np.array(train_X)
	test_X =  np.array(test_X)
	train_Y = np.squeeze(np.array(train_Y))

	train_X, test_X = normalize(train_X, test_X)

	if(opts.train):
		train_func(train_X, train_Y, opts)
	elif(opts.infer):
		infer_func(test_X, opts)


if(__name__ == "__main__"):
	parser = argparse.ArgumentParser(description = "Logistic Regression")
	group = parser .add_mutually_exclusive_group()
	group.add_argument('--train', action = "store_true", default = False, dest = "train", help = "Input --train to Train")
	group.add_argument('--infer', action = "store_true", default = False, dest = "infer", help = "Input --infer to Infer" )
	parser.add_argument('--train_data_path', type = str, default = 'feature/X_train', dest = 'train_data_path', help = "path of training data")
	parser.add_argument('--train_label_path', type = str, default = 'feature/Y_train', dest = 'train_label_path', help = "path of training label")
	parser.add_argument('--test_data_path', type = str, default = 'feature/X_test', dest = 'test_data_path', help = "path of testing data")
	parser.add_argument('--save_dir', type = str, default = 'logistic_params/', dest = 'save_dir', help = "path to save parameters")
	parser.add_argument('--output_dir', type = str, default = 'logistic_output/', dest = 'output_dir', help = "path to save model parameters")
	opts = parser.parse_args()
	main(opts)





