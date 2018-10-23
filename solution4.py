# coding=utf-8

import numpy as np
import zipfile
import scipy.sparse as sp
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from collections import OrderedDict
import gc


def equal_size_bin(m, n):
	quotient = int(m / n)
	remainder = m % n
	if remainder > 0:
		bin_list = [quotient] * (n - remainder) + [quotient + 1] * remainder
	else:
		bin_list = [quotient] * n

	for i in range(1, len(bin_list)):
		bin_list[i] = bin_list[i] + bin_list[i - 1]
	return bin_list

def load_test_data(inzip):
	zips = zipfile.ZipFile(inzip)  # read zip file
	namelist = zips.namelist()  # get name list of zip
	row_test = [];col_test = [];data_test = []
	x_test = []
	row_num_train = 0
	pca = PCA(n_components=3000, iterated_power=1)
	i = 8937;y_lens = 0  # start 8937 end 10948
	# split_iters = [9236,9636,9900,10200,10400,10600,10948]
	split_iters = [9236,9636,9900,10200,10400,10600,10948]
	ii = 0
	maxs = 0 # the max column
	for filename in namelist:
		if filename.endswith("/"):
			continue
		x = zips.read(filename).decode("utf-8")
		sentence_vect = read_x(x)  # get every word or token vector
		for key in sentence_vect.keys():
			if maxs < max(sentence_vect[key]):
				maxs = max(sentence_vect[key])
			for value in sentence_vect[key]:
				row_test.append(row_num_train)
				col_test.append(value)
				data_test.append(1)
			row_num_train += 1
		y_lens += len(sentence_vect)
		if i == split_iters[ii]:
			print('the max column is: ', maxs)
			x_temp = sp.coo_matrix((data_test, (row_test, col_test)), shape=(y_lens, maxs + 1), dtype=np.int8)
			x_temp = x_temp.todense()
			gc.collect()
			print('start process of pca')
			if x_test == []:
				pca = pca.fit(x_temp)  # the speed will be slow, because it training
				x_temp = pca.transform(x_temp)
				x_test = x_temp
			else:
				x_temp = pca.transform(x_temp)  # the speed will be faster
				x_test = np.concatenate((x_test, x_temp), axis=0)
			print('end process of pca')
			y_lens = 0
			row_test = [];col_test = [];data_test = []
			row_num_train = 0
			np.save(str(i) + 'test', x_test)
			ii += 1
			maxs = 0
		i += 1

def load_train_data(inzip):
	zips = zipfile.ZipFile(inzip)  # read zip file
	namelist = zips.namelist()  # get name list of zip
	y_train = []  # split data to train and test
	row_train = [];col_train = [];data_train = []
	x_train = []
	row_num_train = 0
	split_iters = equal_size_bin(8936, 20)
	pca = PCA(n_components=300, iterated_power=1)
	i = 1;ii=0
	label_num = 0
	maxs = 0
	for filename in namelist:
		if filename.endswith(".y") or filename.endswith("/"):
			continue
		x = zips.read(filename).decode("utf-8")
		y = zips.read(filename.split('.')[0] + ".y").decode("utf-8")
		sentence_vect = read_x(x)  # get every word or token vector
		y_list = read_y(y)  # get label of every word or token
		for key in sentence_vect.keys():
			if maxs < max(sentence_vect[key]):
				maxs = max(sentence_vect[key])
			for value in sentence_vect[key]:
				row_train.append(row_num_train)
				col_train.append(value)
				data_train.append(1)
			row_num_train += 1
		for _y in y_list:
			y_train.append([_y])
			label_num += 1
		if i == split_iters[ii]:
			print('the max column is: ', maxs)
			x_train_temp = sp.coo_matrix((data_train, (row_train, col_train)), shape=(label_num, maxs + 1), dtype=np.int8)
			x_train_temp = x_train_temp.todense()
			gc.collect()
			print('start process of pca')
			if x_train == []:
				pca = pca.fit(x_train_temp)
				x_train_temp = pca.transform(x_train_temp)
				x_train = x_train_temp
			else:
				x_train_temp = pca.transform(x_train_temp)
				x_train = np.concatenate((x_train, x_train_temp), axis=0)
			print('end process of pca')
			label_num = 0
			row_train = [];col_train = [];data_train = []
			row_num_train = 0
			np.save(str(i) + 'train', x_train)
			np.save(str(i) + 'label', np.asarray(y_train))
			ii += 1
		i += 1
	return x_train, y_train  # list convert array

def read_x(text):
	sentence_vect = OrderedDict()
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

def read_y(text):
	y_list = []
	for line in text.split("\n"):
		if not line:
			continue
		y_list.append(int(line))
	return y_list

def read_x_vect(text):
    sentence_vect = OrderedDict()
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

def model(x_train, y_train, x_test):
	kernel = 1.0 * RBF(1.0)
	gpc = GaussianProcessClassifier(kernel=kernel, n_jobs=2, max_iter_predict=30)

	m = gpc.fit(x_train, y_train)
	joblib.dump(m, "./model.pkl")
	y_predict = m.predict(x_test) # predict label
	y_predict_log = m.predict_proba(x_test) # predict every label log probability

	return m, y_predict, y_predict_log

def predict_performance(y_test, class_pred, class_logprob):
	correct = np.zeros(y_test.shape[0])
	ltest = np.zeros(y_test.shape[0])
	for i, x in enumerate(y_test):
		correct[i] = y_test[i] == class_pred[i]
		ltest[i] = class_logprob[i, np.argmax(y_test[i, :])]
	er = 1 - correct.mean()
	mnlp = -ltest.mean()
	print("ER: ", er)
	print("MNLP: ", mnlp)
	return er, mnlp

def predict(inzip,m=None,test_data=None):
	zips = zipfile.ZipFile(inzip)  # read zip file
	namelist = zips.namelist()  # get name list of zip
	if m == None:
		m = joblib.load("./model.pkl")
	f = open("./predictions.txt", 'w', encoding='utf-8')
	if test_data == None:
		test_data = np.load('10948test.npy')
	y_pred = m.predict_proba(test_data)
	index = 0
	for filename in namelist:
		if filename.endswith("/"):
			continue
		x = zips.read(filename).decode("utf-8")
		sentence_vect = read_x_vect(x)  # get every word or token vector
		lens = len(sentence_vect)
		for y in y_pred[index:lens]:
			y = ','.join(y)
			f.write(y + "\n")
		f.write("\n")
		index += lens

def split_data(x_train=None,y_train=None):
	from sklearn.model_selection import train_test_split
	if x_train == None:
		x_train, y_train = np.load('8936train.npy'), np.load('8936label.npy')
	X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)
	return X_train, y_train, X_test, y_test

if __name__ == '__main__':

	x_test = load_test_data('conll_test_features.zip')

	x_train,y_train = load_train_data('conll_train.zip')

	x_train, y_train, x_valid, y_valid = split_data(x_train,y_train)

	m, y_predict, y_predict_log = model(x_train, y_train, x_valid)

	predict_performance(y_valid, y_predict_log, y_predict_log)

	predict('conll_test_features.zip',m, x_test)
