# coding=utf-8

import GPy
import numpy as np
import zipfile
import scipy.sparse as sp
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.decomposition import SparsePCA,PCA
import random

def load_train_data(inzip):
	zips = zipfile.ZipFile(inzip)
	namelist = zips.namelist()
	x_matrix = []
	y_matrix = []
	# trian ,valid
	for filename in namelist:
		if filename.endswith(".y") or filename.endswith("/"):
			continue
		x = zips.read(filename).decode("utf-8")
		sentence_vect = {}
		for line in x.split("\n"):
			if not line:
				continue
			line = line.split(" ")
			if line[0] not in sentence_vect:
				vect = [0]*2035523
				vect[int(line[1])] = 1
				sentence_vect[int(line[0])] = vect
			else:
				sentence_vect[int(line[0])][int(line[1])] = 1
		y = zips.read(filename.split('.')[0]+".y").decode("utf-8")
		y_list = []
		for line in y.split("\n"):
			if not line:
				continue
			y_list.append(line)
		for key in sentence_vect.keys():
			x_matrix.append(sentence_vect[key])
			y_matrix.append(y_list[key-1])
	return x_matrix,y_matrix

def load_train_data1(inzip):
    zips = zipfile.ZipFile(inzip) # read zip file
    namelist = zips.namelist() # get name list of zip
    y_train = []; y_test = [] # split data to train and test
    row_train = []; col_train = []; data_train = []
    row_test = []; col_test = []; data_test = []
    row_num_train=0;row_num_test=0
    num = int(len(namelist)/2) # number of set of data
    num_random = list(range(1,num))
    random.shuffle(num_random)
    num_train = num_random[0:int(num*0.9)]  # set about data of 90% as train, 10% data as test
    for filename in namelist:
        if filename.endswith(".y") or filename.endswith("/"):
            continue
        x = zips.read(filename).decode("utf-8")
        y = zips.read(filename.split('.')[0] + ".y").decode("utf-8")

        sentence_vect = read_x_1(x) # get every word or token vector
        y_list = read_y(y) # get label of every word or token

        for key in sentence_vect.keys():
            if int(filename.split("/")[1].split(".")[0]) in num_train:
                for value in sentence_vect[key]:
                    row_train.append(row_num_train)
                    col_train.append(value)
                    data_train.append(1)
                row_num_train += 1
                for _y in y_list:
                    y_train.append([_y])
            else:
                for value in sentence_vect[key]:
                    row_test.append(row_num_test)
                    col_test.append(value)
                    data_test.append(1)
                row_num_test += 1
                for _y in y_list:
                    y_test.append([_y])

    # print(len(y_train[:0]))
    x_train = sp.coo_matrix((data_train,(row_train,col_train)),shape=(len(y_train),100000),dtype=np.int8)
    x_test = sp.coo_matrix((data_test,(row_test,col_test)),shape=(len(y_test),100000),dtype=np.int8)
    # x_train = x_train.todense();x_test = x_test.todense()
    x_train = x_train.toarray(); x_test = x_test.toarray()
    pca = PCA(10000)
    x_train = pca.fit_transform(x_train);x_test = pca.fit_transform(x_test)
    return x_train, np.matrix(y_train), x_test, np.matrix(y_test)  # list convert array
def read_x_1(text):
    sentence_vect = {}
    for line in text.split("\n"):
        if not line:
            continue
        line = line.split(" ")
        if int(line[0]) not in sentence_vect:
            sentence_vect[int(line[0])] = [int(line[1])]
        else:
            temp = sentence_vect[int(line[0])]
            temp.append(int(line[1]))
            sentence_vect[int(line[0])] = temp
    return sentence_vect
def read_x(text):
    sentence_vect = {}
    for line in text.split("\n"):
        if not line:
            continue
        line = line.split(" ")
        if line[0] not in sentence_vect:
            vect = [0] * 2035523
            vect[int(line[1])] = 1
            sentence_vect[int(line[0])] = vect
        else:
            sentence_vect[int(line[0])][int(line[1])] = 1
    return sentence_vect
def read_y(text):
    y_list = []
    for line in text.split("\n"):
        if not line:
            continue
        y_list.append(int(line))
    return y_list
def model_1(x_train, y_train, x_test):
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel)
    # x_train = [[1, 0, 1, 0, 1, 1, 1],
    #      [0, 0, 1, 1, 1, 0, 0],
    #      [1, 1, 1, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 1, 1, 1],
    #      [0, 1, 0, 1, 0, 0, 1]]
    # y_train = [[1], [0], [2], [2], [0]]
    # x_train = np.asarray(x_train)
    # y_train = np.asarray(y_train)
    m = gpc.fit(x_train,y_train)
    # x_test = [[0, 0, 1, 1, 1, 0, 1],
    #         [1, 1, 1, 0, 0, 0, 1]]
    y_predict = m.predict(x_test)
    print(y_predict)
    y_predict_log = m.predict_proba(x_test)
    print(y_predict_log)
    return y_predict,y_predict_log
def predict_performance(y_test, class_pred, class_logprob):
    correct = np.zeros(y_test.shape[0])
    ltest = np.zeros(y_test.shape[0])
    for i, x in enumerate(y_test):
        correct[i] = y_test[i] == class_pred[i]
        ltest[i] = class_logprob[i, np.argmax(y_test[i,:])]
    accuracy = correct.mean()
    loglike = ltest.mean()
    return accuracy, loglike


def model(x_matrix,y_matrix):
	likelihood = GPy.likelihoods.Gaussian()
	module = GPy.models.GPClassification(x_matrix, y_matrix, likelihood=likelihood)



load_train_data('conll_train.zip')
def test():
	x = [[1,0,1,0,1,1,1],
		 [0,0,1,1,1,0,0],
		 [1,1,1,0,0,0,0],
		 [0,0,0,0,1,1,1],
		 [0,1,0,1,0,0,1]]
	# y = [[1],[2],[0],[1],[0]]
	# y = [[1,0],[0,1],[0,1],[1,0],[1,0]]
	y = [[1],[0],[2],[1],[0]]

	# print(GPy.kern)
	x = np.asarray(x)
	y = np.asarray(y)
	likelihood = GPy.likelihoods.LogLogistic()
	module = GPy.models.GPClassification(x,y,likelihood=likelihood)
	t = module.parameters
	print(t)
	print(GPy.likelihoods.Bernoulli.__name__)
