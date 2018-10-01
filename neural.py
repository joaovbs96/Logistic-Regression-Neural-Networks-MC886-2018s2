# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 02 - Neural Network w/ One Hidden Layer
# Tamara Campos - RA 157324
# João Vítor B. Silva - RA 155951

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def sigmoid_derivative(z):
    return z * (1.0 - z)

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def loss(X, y, w):
    m = len(y)
    h = sigmoid(X.dot(w)) # confirmar: dot(x, w)?

    cost = (-1 / m) * np.sum(np.sum(y * np.log(h) + (1 - y) * np.log(1 - h), axis=1))
    # confirmar: soma dupla?

    return cost

def NeuralNetwork(x, y, n, it, vx):
    # step by step: check slide 68
    # step 1 - random init in [0, 1)
    y = np.reshape(y, (y.shape[0], 1))
    weights1 = np.random.rand(n, n)
    weights2 = np.random.rand(n, 1)

    for i in range(it):
        # step 2 - feed forward - slide 41 # TODO: outras funções de ativação
        layer1 = sigmoid(np.dot(x, weights1)) # z = sig(W1x + b1)
        output = sigmoid(np.dot(layer1, weights2)) # y = sig(W2sig(W1x + b1) + b2)

        # step 3 - calculate error
        # TODO

        # step 4 - calculate derivative of error
        # 2(y - y') * z(1 - z) * x
        temp = 2 * (y - output) * sigmoid_derivative(output)
        d_weights2 = np.dot(layer1.T, temp)

        temp = np.dot(temp, weights2.T) * sigmoid_derivative(layer1)
        d_weights1 = np.dot(x.T, temp)

        # steps 5 & 6 - backprop & update the weights with the derivative cost function
        weights1 += d_weights1
        weights2 += d_weights2

    # TODO: optimizer function

    # calculte output value
    layer1 = sigmoid(np.dot(vx, weights1))
    output = sigmoid(np.dot(layer1, weights2))

    return output

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

# redução de dimensionalidade
pca = PCA(.95)
X = pca.fit_transform(X)

# insert bias
m, n = X.shape
temp = np.zeros((m, n + 1))
temp[:, :-1] = X
X = temp

# split dataset into train and validation
trainX, validX = X[10000:], X[:10000]
trainY, validY = Y.iloc[10000:], Y.iloc[:10000]

# Scaler object
scaler = MinMaxScaler(feature_range=(0, 1))    # Between 0 and 1

trainX = scaler.fit_transform(trainX)
validX = scaler.transform(validX)

# m -> number of observations
# n -> number of features
_, n = trainX.shape
it = 200

#One vs all
neural = []
for i in range(10):
    map = {k:v for k, v in zip(range(10), 10*[0])}
    map[i] = 1

    mappedY = trainY['label'].map(map).values

    neural.append(NeuralNetwork(trainX, mappedY, n, it, validX))
    print(i)

results = []
neural = np.asarray(neural).T
for n in neural:
    results.append(np.argmax(n))

print(np.asarray(results).shape)
print(results)

print("F1 Score:" + str(f1_score(validY['label'].values, results, average='micro')))
print("Acuracy: " + str(accuracy_score(validY['label'].values, results)))
