# coding=utf-8

import GPy
import numpy as np
import zipfile

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
