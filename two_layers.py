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
from sklearn.metrics import confusion_matrix
import seaborn as sb

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

def oneHotEncode(y, k):
    return np.eye(k)[y]


def loss(h, y):
    m = len(y)

    cost = (-1.0 / m) * np.sum(np.sum(y * np.log(h + 0.0001) + (1.0 - y) * np.log((1.0 - h) + 0.0001), axis=1))

    return cost

def NeuralNetwork(x, y, it, alpha):
    classes = len(y[0]) # number of classes
    _, n = x.shape # n: number of features
    secondLayerSize = 64
    thirdLayerSize = 64

    # step 1 - random init in [0, 1)
    weights1 = np.random.rand(n, secondLayerSize)
    weights2 = np.random.rand(secondLayerSize, thirdLayerSize)
    weights3 = np.random.rand(thirdLayerSize, classes)

    J = []
    for i in range(it):
        print(i)
        # step 2 - feed forward
        # First activation function
        # z1 = f(W1 * x + b1)
        z1 = np.dot(x, weights1)
        a1 = relu(z1)

        # Second activation function
        # z2 = f(W2 * z1 + b2)
        z2 = np.dot(a1, weights2)
        a2 = relu(z2)

        # Third activation function(softmax)
        # z3 = softmax(W3 * z2 + b3)
        z3 = np.dot(a2, weights3)
        a3 = softmax(z3)

        # step 3 - calculate ouput error
        J.append(loss(a3, y))

        # step 4 - calculate derivative of error
        # step 5 - backpropagate
        # step 6 - update the weights with the derivative cost function
        dz3 = (y - a3)
        dw3 = a2.T.dot(dz3)

        dz2 = dz3.dot(weights3.T) * relu_derivative(a2)
        dw2 = a1.T.dot(dz2)

        dz1 = dz2.dot(weights2.T) * relu_derivative(a1)
        dw1 = x.T.dot(dz1)

        weights3 += dw3 * alpha 
        weights2 += dw2 * alpha 
        weights1 += dw1 * alpha 

    return [weights1, weights2, weights3], J


# MAIN

# execução: main.py [train_data]
# dataset: https://www.dropbox.com/s/qawunrav8ri0sp4/fashion-mnist-dataset.zip

# disable SettingWithCopyWarning warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Read train database
trainName = sys.argv[1]
data = pd.read_csv(trainName)

# read test database
filename = sys.argv[2]
testData = pd.read_csv(filename)

# separate target/classes from set
Y = data.drop(data.columns.values[1:], axis='columns')
X = data.drop('label', axis='columns')

# separate target/classes from test set
testY = testData.drop(testData.columns.values[1:], axis='columns')
testX = testData.drop('label', axis='columns')

# split dataset into train and validation
trainX, validX = X.iloc[12000:], X.iloc[:12000]
trainY, validY = Y.iloc[12000:], Y.iloc[:12000]

# Dimensionality reduction
pca = PCA(.98)
trainX = pca.fit_transform(trainX)
validX = pca.transform(validX)
testX = pca.transform(testX)

# normalize train, validation ans test with the max value in the matrix - monocromatic images
trainX /= 255.0
validX /= 255.0
testX /= 255.0

# target OndeHotEncode and reduction from 3d to 2d matrix
trainY = np.squeeze(oneHotEncode(trainY, len(np.unique(trainY))))

# train neural network
it = 100
alpha = 0.0001 #0.0001
weights, J = NeuralNetwork(trainX, trainY, it, alpha)

# plot graph for GD
plt.plot(J)
plt.ylabel('Função de custo J')
plt.xlabel('Número de iterações')
plt.title('Rede Neural com duas camadas escondidas')
plt.show()

# predict value with validation
z1 = relu(np.dot(testX, weights[0]))
z2 = relu(np.dot(z1, weights[1]))
results = softmax(np.dot(z2, weights[2]))

# get highest probability for each observation
yPred = []
for r in results:
    yPred.append(np.argmax(r))

# Accuracy
print("REDE NEURAL COM 2 CAMADAs ESCONDIDAs - ALPHA 0.0001 - 100 ITERAÇÕES")
print("F1 Score:" + str(f1_score(testY, yPred, average='micro')))

# confusion matrix
cm = confusion_matrix(testY, yPred)
print(cm)

# Heat map
classes = np.unique(testY)
heatMap = sb.heatmap(cm, cmap=sb.color_palette("Blues"))
plt.title("Heat Map Rede Neural 2 Camadas Escondidas")
plt.show()
