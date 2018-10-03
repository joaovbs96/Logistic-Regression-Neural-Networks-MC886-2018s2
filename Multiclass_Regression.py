# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 02 - Logistic Regression
# Tamara Campos - RA 157324
# João Vítor B. Silva - RA 155951

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize as op
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

#------------------------------------------------------#
def oneHotEncode(y, k):
    return np.eye(k)[y]

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def loss(X, y, w):
    m = len(y)
    h = softmax(X.dot(w))

    cost = (-1 / m) * np.sum(np.sum(y * np.log(h) + (1 - y) * np.log(1 - h), axis=1))
    gradient = (1 / m) * np.dot(X.T, (h - y))

    return cost, gradient

    #Optimal theta
def logisticRegressionMulticlass(X, y, w):
    alpha = 0.1
    it = 10000
    J = np.zeros(it)

    for i in range(it):
        J[i], gradientD = loss(X, y, w)
        w = w - (alpha * gradientD)

    return J, w


# MAIN

# execução: main.py [train_data]
# dataset: https://www.dropbox.com/s/qawunrav8ri0sp4/fashion-mnist-dataset.zip

# disable SettingWithCopyWarning warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Read main database
trainName = sys.argv[1]
data = pd.read_csv(trainName)

# separate target/classes from set
Y = data.drop(data.columns.values[1:], axis='columns')
X = data.drop('label', axis='columns')

# split dataset into train and validation
trainX, validX = X.iloc[12000:], X.iloc[:12000]
trainY, validY = Y.iloc[12000:], Y.iloc[:12000]

# Scaler object
scaler = MinMaxScaler(feature_range=(0, 1))    # Between 0 and 1

trainX = scaler.fit_transform(trainX)
validX = scaler.transform(validX)

# insert bias after normalization
trainX = np.insert(trainX, 0, 1, axis=1)
validX = np.insert(validX, 0, 1, axis=1)

k = len(np.unique(trainY))

# (785,10)
w = np.ones([trainX.shape[1],k])

# reduce 3d to 2d matrix
y = np.squeeze(oneHotEncode(trainY, k))

J, w = logisticRegressionMulticlass(trainX, y, w)

# plot graph for GD with regularization
plt.plot(J)
plt.ylabel('Função de custo J')
plt.xlabel('Número de iterações')
plt.title('Regressão Logistica Multinomial')
plt.show()

#Predictions
P = softmax(validX.dot(w)) # probability of each class

results = []
for p in P:
    results.append(np.argmax(p))

print("F1 Score:" + str(f1_score(validY['label'].values, results, average='micro')))
