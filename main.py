# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 02
# Tamara Martinelli de Campos - RA 157324
# João Vítor B. Silva - RA 155951

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return s

# calculates gradient descent
def gradientDescent(x, y, alpha, n, m, it):
    xTran = x.transpose()
    thetas = np.ones(n)
    J = np.zeros(it)

    for i in range(it):
        hypothesis = sigmoid(np.dot(x, thetas.T))

        cost = np.dot(y['label'].values, np.log(hypothesis + 0.001))
        cost += np.dot((1 - y['label'].values), np.log(1 - hypothesis + 0.001))
        J[i] = ((np.sum(cost)/(-1 * m)))

        diff = hypothesis - y['label'].values
        gradient = np.squeeze(np.dot(xTran, diff))/m
        thetas = np.squeeze(thetas - alpha * gradient)

    return J, thetas


## MAIN

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

# insert bias
m, _ = X.shape
X.insert(0, 'bias', np.array(m*[1.0]))

# split dataset into train and validation
trainX, validX = X.iloc[10000:], X.iloc[:10000]
trainY, validY = Y.iloc[10000:], Y.iloc[:10000]

# normalize features
for c in trainX.columns.values:
    if c != 'bias':
        trainX[c] = trainX[c] / 255.0
        validX[c] = validX[c] / 255.0

# calculate cost
m, n = trainX.shape
it = 200
alpha = 0.001

# 10 iterações, cada uma um i é igual a 1 no mapa
J, thetas = [], []
for i in range(10):
    map = {k:v for k, v in zip(range(10), 10*[0])}
    map[i] = 1
    trainY['label'].map(map)

    J_i, thetas_i = gradientDescent(trainX.values, trainY, alpha, n, m, it)
    J.append(J_i)
    thetas.append(thetas_i)

    # plot graph for GD with regularization
    #plt.plot(J[i], 'blue')
    #plt.ylabel('Função de custo J')
    #plt.xlabel('Número de iterações')
    #plt.title('Logistic Regression')
    #plt.show()
