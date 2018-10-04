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
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sb

# Function to transform target data into one hot encode
def oneHotEncode(y, k):
    return np.eye(k)[y]

# Stable softmax function
def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

# Calculate loss and gradient descent
def loss(X, y, w):
    m = len(y)
    h = softmax(X.dot(w))

    cost = (-1 / m) * np.sum(np.sum(y * np.log(h) + (1 - y) * np.log(1 - h), axis=1))
    gradient = (1 / m) * np.dot(X.T, (h - y))

    return cost, gradient

# Perform training
def logisticRegressionMulticlass(X, y, w):
    alpha = 0.1
    it = 1000
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

# read test database
filename = sys.argv[2]
testData = pd.read_csv(filename)

# separate target/classes from train set
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

# Scaler object
scaler = MinMaxScaler(feature_range=(0, 1))    # Between 0 and 1

# normalize train, validation ans test with data found in train
trainX = scaler.fit_transform(trainX)
validX = scaler.transform(validX)
testX = scaler.transform(testX) 

# insert bias after normalization
trainX = np.insert(trainX, 0, 1, axis=1)
validX = np.insert(validX, 0, 1, axis=1)
testX  = np.insert(testX, 0, 1, axis=1)

# Number of classes
numberOfClasses = len(np.unique(trainY))

# Initialize weights
w = np.ones([trainX.shape[1],numberOfClasses])

# reduce 3d to 2d matrix
y = np.squeeze(oneHotEncode(trainY, numberOfClasses))

# execute multinomial regression
J, w = logisticRegressionMulticlass(trainX, y, w)

# plot graph for GD with regularization
plt.plot(J)
plt.ylabel('Função de custo J')
plt.xlabel('Número de iterações')
plt.title('Regressão Logistica Multinomial. Alpha = 0.1')
plt.show()

#Predictions
P = softmax(testX.dot(w)) # probability of each class

results = []
for p in P:
    results.append(np.argmax(p))

# Accuracy
print("REGRESSÃO LOGISTICA MULTINOMIAL - ALPHA 0.1 - 1000 ITERAÇÕES")
print("F1 Score:" + str(f1_score(testY['label'].values, results, average='micro')))

# confusion matrix
cm = confusion_matrix(testY['label'].values, results)
print(cm)

# Heat map
classes = np.unique(trainY)
heatMap = sb.heatmap(cm, cmap=sb.color_palette("Blues"))
plt.title("Heat Map Regressão Logistica Multinomial")
plt.show()
