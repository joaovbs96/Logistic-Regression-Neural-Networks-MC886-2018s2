import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Plots
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score  
#Advanced optimization
from scipy import optimize as op

from sklearn.preprocessing import MinMaxScaler
import sys

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

#Regularized cost function
def costFunction(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))

    return (1 / m) * (-y.T.dot(np.log(h + 0.001)) - (1 - y).T.dot(np.log(1 - h + 0.001)))

#Regularized gradient function
def gradient(theta, X, y):
    h = sigmoid(X.dot(theta))

    return ((1 / m) * X.T.dot(h - y))

#Optimal theta 
def logisticRegression(X, y, theta):

    alpha = 0.001
    it = 200
    J = np.zeros(it)

    for i in range(it):
        J[i] = costFunction(theta, X, y)
        gradientD = gradient(theta, X, y)
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

# insert bias
m, _ = X.shape
X.insert(0, 'bias', np.array(m*[1.0]))

# split dataset into train and validation
trainX, validX = X.iloc[10000:], X.iloc[:10000]
trainY, validY = Y.iloc[10000:], Y.iloc[:10000]

# Scaler object
scaler = MinMaxScaler(feature_range=(0, 1))    # Between 0 and 1

trainX = scaler.fit_transform(trainX)
validX = scaler.transform(validX)

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
plt.plot(J[0])
plt.ylabel('Função de custo J')
plt.xlabel('Número de iterações')
plt.title('Logistic Regression')
plt.show()
#

#Predictions
P = sigmoid(validX.dot(all_theta.T)) #probability for each flower
#print(P)
print(P)
#p = [validY[np.argmax(P[i, :])] for i in range(m)]
#p = [Species[np.argmax(P[i, :])] for i in range(trainX.shape[0])]

#print("Test Accuracy ", accuracy_score(validY, p) * 100 , '%')

