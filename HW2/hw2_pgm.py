import numpy as np
import re
import math
import argparse
import os


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

def valid(X, Y, mu1, mu2, shared_sigma, N1, N2):

	data_size = X.shape[0]
	y_ = predict(X, mu1, mu2, shared_sigma, N1, N2)
	result = (y_==Y)

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

def get_para(train_X, train_Y):

	train_set_size = train_X.shape[0]
	dim = train_X.shape[1]
	mu1 = np.zeros((dim,))
	mu2 = np.zeros((dim,))
	cnt1 = 0
	cnt2 = 0
	for i in range(train_set_size):
		if(train_Y[i] == 1):
			mu1 += train_X[i]
			cnt1 += 1
		else:
			mu2 += train_X[i]
			cnt2 += 1
	mu1 = mu1/cnt1
	mu2 = mu2/cnt2

	sigma1 = np.zeros((dim, dim))
	sigma2 = np.zeros((dim, dim))
	cnt1 = 0
	cnt2 = 0
	#print(mu1)
	#print(mu2)
	for i in range(train_set_size):
		if(train_Y[i] == 1):
			sigma1 += np.dot(np.transpose([train_X[i] - mu1]), ([train_X[i] - mu1]))
			cnt1 += 1
		else:
			sigma2 += np.dot(np.transpose([train_X[i] - mu2]), ([train_X[i] - mu2]))
			cnt2 += 1

	
	sigma1 /= cnt1
	sigma2 /= cnt2

	shared_sigma = (float(cnt1)/train_set_size)*sigma1 + (float(cnt2)/train_set_size)*sigma2
	N1 = cnt1
	N2 = cnt2

	print("=====saving parameters=====")
	if(not os.path.exists(opts.save_dir)):
		os.mkdir(opts.save_dir)
	save_dic = {"mu1": mu1, "mu2": mu2, "shared_sigma": shared_sigma, "N1": [N1], "N2": [N2]}

	for key in save_dic:
		#print(key+": "+str(save_dic[key]))
		np.savetxt(os.path.join(opts.save_dir, ('%s' % key)), save_dic[key])

	return mu1, mu2, shared_sigma, N1, N2


def predict(X, mu1, mu2, shared_sigma, N1, N2):
	sigma_inv = np.linalg.inv(shared_sigma)
	w = np.dot((mu1 - mu2), sigma_inv)
	X = np.transpose(X)
	b = (-0.5) * np.dot(np.dot(mu1, sigma_inv), mu1) + (0.5) * np.dot(np.dot(mu2, sigma_inv), mu2) + np.log(float(N1)/N2)
	a = np.dot(w, X) + b
	y = sigmoid(a)
	y = np.around(y)
	return y


def train_func(train_X, train_Y, opts ):


	train_X, train_Y, valid_X, valid_Y = split_train_valid(train_X, train_Y, 0.9)

	mu1, mu2, shared_sigma, N1, N2 = get_para(train_X, train_Y)

	valid(valid_X, valid_Y, mu1, mu2, shared_sigma, N1, N2)





def infer_func(test_X, opts):
	print("=====load the parameters=====")
	mu1 = np.loadtxt(os.path.join(opts.save_dir, "mu1"))
	mu2 = np.loadtxt(os.path.join(opts.save_dir, "mu2"))
	shared_sigma = np.loadtxt(os.path.join(opts.save_dir, "shared_sigma"))
	N1 = np.loadtxt(os.path.join(opts.save_dir, "N1"))
	N2 = np.loadtxt(os.path.join(opts.save_dir, "N2"))

	y = predict(test_X, mu1, mu2, shared_sigma, N1, N2)

	print("=====save the result into ans.csv=====")
	if(not os.path.exists(opts.output_dir)):
		os.mkdir(opts.output_dir)
	output_path = os.path.join(opts.output_dir, "ans.csv")
	with open(output_path, "w+") as f:
		f.write("id,label\n")
		count = 1
		for value in y:
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
	group = parser.add_mutually_exclusive_group()
	group.add_argument('--train', action = "store_true", default = False, dest = "train", help = "Input --train to Train")
	group.add_argument('--infer', action = "store_true", default = False, dest = "infer", help = "Input --infer to Infer" )
	parser.add_argument('--train_data_path', type = str, default = 'feature/X_train', dest = 'train_data_path', help = "path of training data")
	parser.add_argument('--train_label_path', type = str, default = 'feature/Y_train', dest = 'train_label_path', help = "path of training label")
	parser.add_argument('--test_data_path', type = str, default = 'feature/X_test', dest = 'test_data_path', help = "path of testing data")
	parser.add_argument('--save_dir', type = str, default = 'pgm_params/', dest = 'save_dir', help = "path to save parameters")
	parser.add_argument('--output_dir', type = str, default = 'pgm_output/', dest = 'output_dir', help = "path to save model parameters")
	opts = parser.parse_args()
	main(opts)
