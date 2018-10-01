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

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# cost function
def loss(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))

    cost = (1 / m) * np.sum((-y.T.dot(np.log(h + 0.001)) - (1 - y).T.dot(np.log(1 - h + 0.001))))
    gradient = ((1 / m) * X.T.dot(h - y))

    return cost, gradient

def logisticRegression(X, y, theta):

    alpha = 0.1
    it = 1000
    J = np.zeros(it)

    for i in range(it):
        J[i], gradientD = loss(theta, X, y)
        theta = theta - (alpha * gradientD)

    return J, theta

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

k = 10
m, n = trainX.shape

#Training
all_theta = np.zeros((k, n))

#One vs all
J = []
for i in range(10):
    map = {k:v for k, v in zip(range(10), 10*[0])}
    map[i] = 1

    mappedY = trainY['label'].map(map).values

    J_it, optTheta = logisticRegression(trainX, mappedY,  np.zeros(n))
    J.append(J_it)
    all_theta[i] = optTheta

# plot graph for GD with regularization
# plt.plot(J[0])
# plt.ylabel('Função de custo J')
# plt.xlabel('Número de iterações')
# plt.title('Logistic Regression')
# plt.show()
#

#Predictions
P = sigmoid(validX.dot(all_theta.T)) # probability of each class

results = []
for p in P:
    results.append(np.argmax(p))

"""plt.scatter(results, range(len(results)), color='blue')
plt.scatter(validY['label'].values, range(len(validY['label'].values)), color='red')
plt.ylabel('Classe Predita')
plt.xlabel('Observação')
plt.title('Predição vs Real')
plt.show()"""

print("F1 Score:" + str(f1_score(validY['label'].values, results, average='micro')))
print("Acuracy: " + str(accuracy_score(validY['label'].values, results)))