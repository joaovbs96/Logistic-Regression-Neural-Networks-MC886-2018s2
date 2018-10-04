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



J_it, optTheta = logisticRegression(trainX, mappedY,  np.zeros(n))
J.append(J_it)
all_theta[i] = optTheta

# plot graph for GD with regularization
plt.plot(J[0])
plt.ylabel('Função de custo J')
plt.xlabel('Número de iterações')
plt.title('Logistic Regression')
plt.show()



