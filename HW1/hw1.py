import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt 

lamda = 0.001

#feature 是所有空气质量参数
def train_func_1(n_hour):
	#init the training data
	trainData = []
	for i in range(18):
		trainData.append([])

	n_row = 0
	with open("train.csv", "r", encoding = "big5") as f:
		rows = csv.reader(f, delimiter = ",")
		rows = list(rows)
		for row in rows:
			if(n_row != 0):
				for i in range(3,27):
					if(row[i] != "NR"):
						trainData[(n_row-1)%18].append(float(row[i]))
					else:
						trainData[(n_row-1)%18].append(float(0))
			n_row += 1

	train_x = []
	train_y = []
	for mon in range(12):
		for wid in range(480-n_hour):
			train_x.append([])
			for n in range(18):
				for j in range(n_hour):
					train_x[(480-n_hour)*mon+wid].append(trainData[n][480*mon+wid+j])

			train_y.append(trainData[9][480*mon+wid+n_hour])


	train_x = np.array(train_x)
	train_y = np.array(train_y)
	# print(train_x.shape)
	# print(n_hour)
	train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x), axis=1)
	w = np.zeros(len(train_x[0]))

	learning_rate = 100
	x_t = train_x.transpose()
	s_gra = np.zeros(len(train_x[0]))

	cost_log = [[],[]]
	for i in range(1000):
		hypo = np.dot(train_x, w)
		loss = hypo - train_y
		cost = np.sum((loss**2)) / len(train_x)
		cost_a = math.sqrt(cost)
		gra = np.dot(x_t, loss) + 2 * lamda * w
		s_gra += gra**2
		ada = np.sqrt(s_gra)
		w = w - learning_rate * gra/ada
		cost_log[0].append(i)
		cost_log[1].append(cost_a)
		print("iteration: %d | Cost: %f" % (i, cost_a))

	np.save("model_a.npy", w)
	return cost_log

#feature 是单一的PM2.5
def train_func_2(n_hour):
	#init the training data
	trainData = []
	for i in range(1):
		trainData.append([])

	n_row = 0
	with open("train.csv", "r", encoding = "big5") as f:
		rows = csv.reader(f, delimiter = ",")
		rows = list(rows)
		for row in rows:
			if(n_row != 0 and row[2]=="PM2.5"):
				for i in range(3,27):
					if(row[i] != "NR"):
						trainData[0].append(float(row[i]))
					else:
						trainData[0].append(float(0))
			n_row += 1

	train_x = []
	train_y = []
	for mon in range(12):
		for wid in range(480-n_hour):
			train_x.append([])
			for n in range(1):
				for j in range(n_hour):
					train_x[(480-n_hour)*mon+wid].append(trainData[n][480*mon+wid+j])

			train_y.append(trainData[0][480*mon+wid+n_hour])


	train_x = np.array(train_x)
	train_y = np.array(train_y)

	train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x), axis=1)
	w = np.zeros(len(train_x[0]))

	learning_rate = 10
	x_t = train_x.transpose()
	s_gra = np.zeros(len(train_x[0]))

	cost_log = [[],[]]
	for i in range(1000):
		hypo = np.dot(train_x, w)
		loss = hypo - train_y
		cost = np.sum((loss**2)) / len(train_x)
		cost_a = math.sqrt(cost)
		gra = np.dot(x_t, loss) + 2 * lamda * w
		s_gra += gra**2
		ada = np.sqrt(s_gra)
		w = w - learning_rate * gra/ada
		cost_log[0].append(i)
		cost_log[1].append(cost_a)
		print("iteration: %d | Cost: %f" % (i, cost_a))

	np.save("model_b.npy", w)
	return cost_log


#feature 是所有空气质量参数
def test_func_1():

	w = np.load("model.npy")
	testData = []
	n_row = 0
	n_set = 0
	with open("test.csv", "r", encoding = "big5") as f:
		rows = csv.reader(f, delimiter = ",")
		rows = list(rows)
		for row in rows:
			if(n_row%18==0):
				testData.append([])
				n_set += 1
			for i in range(2,11):
				if(row[i]!="NR"):
					testData[n_set-1].append(float(row[i]))
				else:
					testData[n_set-1].append(float(0))
			n_row+=1

	testData = np.array(testData)
	test_x = np.concatenate((np.ones((testData.shape[0],1)),testData), axis=1)

	test_hypo = np.dot(test_x, w)

	ans = []
	for i in range(test_hypo.shape[0]):
		ans.append(["id_"+str(i)])
		ans[i].append(test_hypo[i])
	print(len(ans))
	print(len(ans[0]))
	fileName = "predict.csv"
	f = open(fileName, "w+")
	csvWriter = csv.writer(f, delimiter = ",", lineterminator = "\n")
	csvWriter.writerow(["id", "value"])
	for i in range(len(ans)):
		csvWriter.writerow(ans[i])
	f.close()








log1 = train_func_1(8) #feature太多容易overfitting 
log2 = train_func_2(8) #feature只有PM2.5，更容易拟合

#plt.subplot(2,1,1)
plt.title("train_1")
plt.plot(log1[0][2:], log1[1][2:], color = "orange")
plt.plot(log2[0][2:], log2[1][2:], color = "green")
plt.xlabel("iteration")
plt.ylabel("cost")
plt.show()



#test_func()

