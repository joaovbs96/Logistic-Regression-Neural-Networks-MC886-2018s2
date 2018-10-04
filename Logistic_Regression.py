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
from sklearn.metrics import confusion_matrix
import seaborn as sb
from sklearn.decomposition import PCA

# sigmoid function
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# cost function and gradient descent
def loss(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))

    # 0.001 was added to avoid cases where h == 0, because log(0) == nan
    cost = (1 / m) * np.sum((-y.T.dot(np.log(h + 0.001)) - (1 - y).T.dot(np.log(1 - h + 0.001))))
    gradient = ((1 / m) * X.T.dot(h - y))

    return cost, gradient

# Perform trainining
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

k = 10
m, n = trainX.shape

#Training
all_theta = np.zeros((k, n))

#One vs all
J = []
for i in range(10):
    # transform tareget data into 0 or 1
    # 1 for the class od interest and 0 for the others
    map = {k:v for k, v in zip(range(10), 10*[0])}
    map[i] = 1

    mappedY = trainY['label'].map(map).values

    J_it, optTheta = logisticRegression(trainX, mappedY,  np.zeros(n))
    J.append(J_it)
    all_theta[i] = optTheta

# plot graph for GD with regularization
#plt.plot(J)
#plt.ylabel('Função de custo J')
#plt.xlabel('Número de iterações')
#plt.title('Regressão Logistica One-vs-All. Alpha 0.1')
#plt.show()
#

# ============== VALIDATION

#Predictions
PValid = sigmoid(validX.dot(all_theta.T)) # probability of each class

resultsValid = []
for p in PValid:
    resultsValid.append(np.argmax(p))

print("VALIDAÇÃO ----> REGRESSÃO LOGISTICA ONE-VS-ALL - ALPHA 0.1 - 1000 ITERAÇÕES")
print("F1 Score:" + str(f1_score(validY, resultsValid, average='micro')))

# ============= TEST

#Predictions
P = sigmoid(testX.dot(all_theta.T)) # probability of each class

results = []
for p in P:
    results.append(np.argmax(p))

"""plt.scatter(results, range(len(results)), color='blue')
plt.scatter(validY['label'].values, range(len(validY['label'].values)), color='red')
plt.ylabel('Classe Predita')
plt.xlabel('Observação')
plt.title('Predição vs Real')
plt.show()"""

# Accuracy
print("REGRESSÃO LOGISTICA ONE-VS-ALL - ALPHA 0.1 - 1000 ITERAÇÕES")
print("F1 Score:" + str(f1_score(testY, results, average='micro')))

# confusion matrix
cm = confusion_matrix(testY, results)
print(cm)

# Heat map
classes = np.unique(testY)
heatMap = sb.heatmap(cm, cmap=sb.color_palette("Blues"))
plt.title("Heat Map Regressão Logistica One vs All")
plt.show()