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
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# =========================================

def leaky_relu_derivative(z):
    def do_leaky_deriv(x):
        if x > 0:
            return 1
        else:
            return 0.01

    relufunc = np.vectorize(do_leaky_deriv)
    return relufunc(z)


def leaky_relu(z):
    def do_leaky(x):
        if x > 0:
            return x
        else:
            return 0.01 * x

    relufunc = np.vectorize(do_leaky)
    return relufunc(z)

# =========================================

def relu_derivative(z):
    def do_relu_deriv(x):
        if x > 0:
            return 1
        else:
            return 0

    relufunc = np.vectorize(do_relu_deriv)
    return relufunc(z)


def relu(z):
    maxValue = z.max()
    def do_relu(x):
        return max(0, x) / maxValue

    relufunc = np.vectorize(do_relu)
    return relufunc(z)

# =========================================

def tanh_derivative(z):
    return 1.0 - (tanh(z) * tanh(z))


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

# =========================================

def sigmoid_derivative(z):
    return z * (1.0 - z)


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# =========================================

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

# =========================================


def oneHotEncode(y, k):
    return np.eye(k)[y]


def loss(h, y):
    m = len(y)

    cost = (-1 / m) * np.sum(np.sum(y * np.log(h + 0.0001) + (1. - y) * np.log((1 - h) + 0.0001), axis=1))

    return cost

def NeuralNetwork(x, y, n, it, vx, numberOfClasses):
    secondLayerSize = 64

    # step by step: check slide 68
    # step 1 - random init in [0, 1)
    weights1 = np.random.rand(n, secondLayerSize)
    weights2 = np.random.rand(secondLayerSize, numberOfClasses)

    J = []
    for i in range(it):
        # step 2 - feed forward - slide 41

        # hidden layer with relu
        layer1 = relu(np.dot(x, weights1)) # z = sig(W1x + b1)

        # output layer with softmax
        output = softmax(np.dot(layer1, weights2)) # y = sig(W2sig(W1x + b1) + b2)

        # step 3 - calculate error
        J.append(loss(output, y))

        # step 4 - calculate derivative of error
        # the last layer don't use activation function derivative
        d_weights2 = (y - output)

        #temp = np.dot(temp, weights2.T) * sigmoid_derivative(layer1)
        #d_weights1 = np.dot(x.T, temp)
        d_weights1 = d_weights2.dot(weights2.T) * relu_derivative(layer1)

        # steps 5 & 6 - backprop & update the weights with the derivative cost function
        weights2 += layer1.T.dot(d_weights2) * 0.0001
        weights1 += x.T.dot(d_weights1) * 0.0001

    # calculte output value
    layer1 = relu(np.dot(vx, weights1))
    output = softmax(np.dot(layer1, weights2))

    return output, J


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

# split dataset into train and validation
trainX, validX = X[10000:], X[:10000]
trainY, validY = Y.iloc[10000:], Y.iloc[:10000]

# normalization 
trainX /= 255.0
validX /= 255.0

# m -> number of observations
# n -> number of features
_, n = trainX.shape
it = 100

numberOfClasses = len(np.unique(trainY))

# reduce 3d to 2d matrix
y = np.squeeze(oneHotEncode(trainY, numberOfClasses))

result, J = NeuralNetwork(trainX, y, n, it, validX, numberOfClasses)

# plot graph for GD with regularization
plt.plot(J)
plt.ylabel('Função de custo J')
plt.xlabel('Número de iterações')
plt.title('Rede Neural com uma camada escondida')
plt.show()

results = []
for r in result:
    results.append(np.argmax(r))

#print("iterações: ", str(it))
#print("alpha: 0.0001")
print("F1 Score:" + str(f1_score(validY['label'].values, results, average='micro')))

