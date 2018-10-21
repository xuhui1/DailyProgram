# coding=utf-8

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
    num = int(len(namelist)/2) # number of set of data

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
        if i%10==0:
            lens += len(y_list)
            x_train_temp = sp.coo_matrix((data_train,(row_train,col_train)),shape=(lens,2035523),dtype=np.int8)
            x_train_temp = x_train_temp.todense()
            gc.collect()
            x_train_temp = pca.fit_transform(x_train_temp)
            if x_train==[]:
                x_train = x_train_temp
            else:
                np.concatenate((x_train, x_train_temp), axis=0)
            i = 1;lens = 0
            row_train = [];col_train = [];data_train = []
            row_num_train = 0
            print(1)
        else:
            i +=1
            lens += len(y_list)
        if i%1000==0:
            np.save(str(i)+'_data.npz', x_train)
    # x_train = x_train.todense();x_test = x_test.todense()
    # pca = PCA(n_components=300)
    # x_train = pca.fit_transform(x_train);x_test = pca.fit_transform(x_test)
    x_train_temp = sp.coo_matrix((data_train, (row_train, col_train)), shape=(lens, 2035523), dtype=np.int8)
    x_train_temp = x_train_temp.todense()
    gc.collect()
    x_train_temp = pca.fit_transform(x_train_temp)
    if x_train == []:
        x_train = x_train_temp
    else:
        np.concatenate((x_train, x_train_temp), axis=0)
    np.save('data.npz',x_train)
    return x_train, np.matrix(y_train), np.matrix(y_test)  # list convert array

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

def predict1(inzip):
    zips = zipfile.ZipFile(inzip)  # read zip file
    namelist = zips.namelist()  # get name list of zip

    m = joblib.load("./model.pkl")
    f = open("./predictions.txt",'w',encoding='utf-8')
    x_test_temp = []
    x_test = []
    i = 1;lens = 0
    pca = PCA(n_components=100)
    sen_len = []
    for filename in namelist:
        if filename.endswith("/"):
            continue
        x = zips.read(filename).decode("utf-8")
        sentence_vect = read_x_vect(x)  # get every word or token vector
        sen_len.append(len(sentence_vect))
        # x_test = []
        for key in sentence_vect.keys():
            x_test_temp.append(sentence_vect[key])
        if lens>170:
            x_test_temp = pca.fit_transform(x_test_temp)
            if x_test == []:
                x_test = x_test_temp
            else:
                np.concatenate((x_test, x_test_temp), axis=0)
        else:
            lens += len(sentence_vect)
    np.save('test',x_test)
    y_predict = m.predict_proba(x_test)
    for y in sen_len:
        y = ','.join(y)
        f.write(y+"\n")
        f.write("\n")

x = np.load('1_data.npy')
y=np.load('label.npy')
print(x)
