# coding=utf-8

import GPy
import numpy as np
import zipfile
import random
import scipy.sparse as sp
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.decomposition import SparsePCA,PCA
from sklearn.externals import joblib
from collections import OrderedDict
import gc

def load_train_data(inzip):
    zips = zipfile.ZipFile(inzip) # read zip file
    namelist = zips.namelist() # get name list of zip
    y_train = []; y_test = [] # split data to train and test
    row_train = []; col_train = []; data_train = []
    x_train = []
    row_num_train=0

    pca = PCA(n_components=100)
    i = 1;lens=0
    for filename in namelist:
        # y_train
        if filename.endswith(".y") or filename.endswith("/"):
            continue
        x = zips.read(filename).decode("utf-8")
        y = zips.read(filename.split('.')[0] + ".y").decode("utf-8")

        sentence_vect = read_x_1(x) # get every word or token vector
        y_list = read_y(y) # get label of every word or token

        for key in sentence_vect.keys():

            for value in sentence_vect[key]:
                row_train.append(row_num_train)
                col_train.append(value)
                data_train.append(1)
            row_num_train += 1
            for _y in y_list:
                y_train.append([_y])
        if lens>170:
            lens += len(y_list)
            x_train_temp = sp.coo_matrix((data_train,(row_train,col_train)),shape=(lens,2035523),dtype=np.int8)
            x_train_temp = x_train_temp.todense()
            gc.collect()
            x_train_temp = pca.fit_transform(x_train_temp)
            if x_train==[]:
                x_train = x_train_temp
            else:
                # x_train = np.load('1_data.npy')
                np.concatenate((x_train, x_train_temp), axis=0)
            lens = 0
            row_train = [];col_train = [];data_train = []
            row_num_train = 0
            print(1)
            np.save('1_data', x_train)
            np.save('label',np.asarray(y_train))
            # x_train = []
        else:
            lens += len(y_list)
        i += 1
        if i%1000==0:
            np.save(str(i) + '_data', x_train)

    # x_train = x_train.todense();x_test = x_test.todense()
    # pca = PCA(n_components=300)
    # x_train = pca.fit_transform(x_train);x_test = pca.fit_transform(x_test)
    x_train_temp = sp.coo_matrix((data_train, (row_train, col_train)), shape=(lens, 2035523), dtype=np.int8)
    x_train_temp = x_train_temp.todense()
    gc.collect()
    x_train_temp = pca.fit_transform(x_train_temp)
    x_train = np.load('1_data.npy')
    np.concatenate((x_train, x_train_temp), axis=0)
    np.save('data',x_train)

    return x_train, y_train, x_test # list convert array


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

    x_train = sp.coo_matrix((data_train,(row_train,col_train)),shape=(len(y_train),100000),dtype=np.int8)
    x_test = sp.coo_matrix((data_test,(row_test,col_test)),shape=(len(y_test),100000),dtype=np.int8)
    x_train = x_train.todense();x_test = x_test.todense()
    # pca = PCA(n_components=300)
    # x_train = pca.fit_transform(x_train);x_test = pca.fit_transform(x_test)
    return x_train, np.matrix(y_train), x_test, np.matrix(y_test)  # list convert array

def read_x_1(text):
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

def read_y(text):
    y_list = []
    for line in text.split("\n"):
        if not line:
            continue
        y_list.append(int(line))
    return y_list

def model1(x_train, y_train, x_test, y_test):
    likelihood = GPy.likelihoods.Gaussian()
    model = GPy.models.GPClassification(x_train, y_train, likelihood=likelihood)
    y_predict = model.predict(x_test)
    print(y_predict)

def model_1(x_train, y_train, x_test):
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel)

    m = gpc.fit(x_train,y_train)
    joblib.dump(m,"./model.pkl")

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
    er =1 - correct.mean()
    mnlp = -ltest.mean()
    return er, mnlp


def train(inzip):
    zips = zipfile.ZipFile(inzip)  # read zip file
    namelist = zips.namelist()  # get name list of zip
    y_train = [];y_test = []  # split data to train and test
    row_train = [];col_train = [];data_train = []
    row_test = [];col_test = [];data_test = []
    row_num_train = 0;row_num_test = 0
    num = int(len(namelist) / 2)  # number of set of data
    for filename in namelist[0:int(num * 0.9)]:
        if filename.endswith(".y") or filename.endswith("/"):
            continue
        x = zips.read(filename).decode("utf-8")
        y = zips.read(filename.split('.')[0] + ".y").decode("utf-8")

        sentence_vect = read_x_1(x)  # get every word or token vector
        y_list = read_y(y)  # get label of every word or token

        for key in sentence_vect.keys():
            for value in sentence_vect[key]:
                row_train.append(row_num_train)
                col_train.append(value)
                data_train.append(1)
            row_num_train += 1
            for _y in y_list:
                y_train.append([_y])

def predict(inzip):
    zips = zipfile.ZipFile(inzip)  # read zip file
    namelist = zips.namelist()  # get name list of zip

    m = joblib.load("./model.pkl")
    f = open("./predictions.txt",'w',encoding='utf-8')
    for filename in namelist:
        if filename.endswith("/"):
            continue
        x = zips.read(filename).decode("utf-8")
        sentence_vect = read_x_vect(x)  # get every word or token vector
        x_test = []
        for key in sentence_vect.keys():
            x_test.append(sentence_vect[key])
        y_predict = m.predict(x_test)
        for y in y_predict:
            y = ','.join(y)
            f.write(y+"\n")
        f.write("\n")

def predict1(inzip):
    zips = zipfile.ZipFile(inzip)  # read zip file
    namelist = zips.namelist()  # get name list of zip

    m = joblib.load("./model.pkl")
    f = open("./predictions.txt",'w',encoding='utf-8')
    for filename in namelist:
        if filename.endswith("/"):
            continue
        x = zips.read(filename).decode("utf-8")
        sentence_vect = read_x_vect(x)  # get every word or token vector
        x_test = []
        for key in sentence_vect.keys():
            x_test.append(sentence_vect[key])
        y_predict = m.predict(x_test)
        for y in y_predict:
            y = ','.join(y)
            f.write(y+"\n")
        f.write("\n")

x_train, y_train, x_test, y_test = load_train_data('conll_train.zip')
#
# model(x_train, y_train, x_test, y_test)
y_predict,y_predict_log = model_1(x_train, y_train, x_test)
# y_test=[[1],[2]]
# y_test = np.asarray(y_test)
predict_performance(y_test,y_predict,y_predict_log)

def test():
    x = [[1, 2],
         [0, 1, 3],
         [2],
         [0, 1, 2],
         [0]]
    import scipy.sparse as sp
    x = sp.csr_matrix(x)
    x_test = [[1],
              [0, 1]]
    x_test = sp.csr_matrix(x_test)
    # y = [[1],[0],[1],[1],[0]]
    y = [[1], [0], [2], [1], [0]]

    x = np.asarray(x)
    x_test = np.asarray(x_test)
    y = np.asarray(y)

    # likelihood = GPy.likelihoods.Exponential()
    likelihood = GPy.likelihoods.Gaussian()
    # likelihood = GPy.likelihoods.Poisson()
    # likelihood = GPy.likelihoods.StudentT()
    m = GPy.models.SparseGPClassification(x, y, likelihood=likelihood)

    print(m.predict(x_test))
    # print(m.predict_jacobian(x_test))
    print(m.predict_magnification(x_test))
    print(m.predict_noiseless(x_test))

# test()
