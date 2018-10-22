# coding=utf-8

import numpy as np
import zipfile
import random
import scipy.sparse as sp
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from collections import OrderedDict

def load_train_data(inzip):
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

        sentence_vect = read_x_site(x) # get every word or token vector
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

    x_train = sp.coo_matrix((data_train,(row_train,col_train)),shape=(len(y_train),2035523),dtype=np.int8)
    x_test = sp.coo_matrix((data_test,(row_test,col_test)),shape=(len(y_test),2035523),dtype=np.int8)
    x_train = x_train.todense();x_test = x_test.todense()
    pca = PCA(n_components=100)
    x_train = pca.fit_transform(x_train);x_test = pca.fit_transform(x_test)
    return x_train, np.matrix(y_train), x_test, np.matrix(y_test)  # list convert array

def read_x_site(text):
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

def train(x_train, y_train):
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel)

    m = gpc.fit(x_train,y_train)
    joblib.dump(m,"./model.pkl")
    return m

def test(x_test,m=None):
    if m==None:
        m = joblib.load("./model.pkl")
    y_predict = m.predict(x_test)
    y_predict_log = m.predict_proba(x_test)
    return y_predict,y_predict_log

def predict_performance(y_test, y_pred, y_logprob):
    correct = np.zeros(y_test.shape[0])
    ltest = np.zeros(y_test.shape[0])
    for i, x in enumerate(y_test):
        correct[i] = y_test[i] == y_pred[i]
        ltest[i] = y_logprob[i, np.argmax(y_test[i,:])]
    er =1 - correct.mean()
    mnlp = -ltest.mean()
    print("The value of ER: ",er)
    print("The value of MNLP: ",mnlp)
    return er, mnlp

def predict(inzip):
    zips = zipfile.ZipFile(inzip)  # read zip file
    namelist = zips.namelist()  # get name list of zip
    x_test = []
    m = joblib.load("./model.pkl")
    f = open("./predictions.txt",'w',encoding='utf-8')
    sentence_len = []
    sentence_len.append(0)
    pca = PCA(n_components=3000)
    for filename in namelist:
        if filename.endswith("/"):
            continue
        x = zips.read(filename).decode("utf-8")
        sentence_vect = read_x_vect(x)  # get every word or token vector
        sentence_len.append(len(sentence_vect))
        for key in sentence_vect.keys():
            x_test.append(sentence_vect[key])
    x_test = np.asarray(x_test)
    x_test = pca.fit_transform(x_test)
    y_predict = m.predict_proba(x_test)
    for i in range(1,len(sentence_len)):
        sentence_len[i] = sentence_len[i-1]+sentence_len[i]
        for y in y_predict[sentence_len[i-1]:sentence_len[i]]:
            y = ','.join(y)
            f.write(y+"\n")
        f.write("\n")

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_train_data('conll_train.zip')
    m = train(x_train, y_train)
    y_predict, y_predict_log = test(x_test,m)
    predict_performance(y_test,y_predict,y_predict_log)
    predict('conll_test_features.zip')
