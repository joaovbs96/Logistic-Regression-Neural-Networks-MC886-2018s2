# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 02 - Logistic Regression
# Tamara Campos - RA 157324
# João Vítor B. Silva - RA 155951

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# =========================================

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
def logisticRegression(X, y, theta, it, alpha):
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

it = 1000
# tested alphas: 0.1, 0.01, 0.001 - Better: 0.1
alphas = [0.1]

for alpha in alphas:
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

        J_it, optTheta = logisticRegression(trainX, mappedY,  np.zeros(n), it, alpha)
        J.append(J_it)
        all_theta[i] = optTheta

    # plot graph for GD with regularization
    #plt.plot(J)
    #plt.ylabel('Função de custo J')
    #plt.xlabel('Número de iterações')
    #plt.title('Regressão Logistica One-vs-All. Alpha:', str(alpha))
    #plt.show()
    #

    # ============== VALIDATION

    #Predictions
    PValid = sigmoid(validX.dot(all_theta.T)) # probability of each class

    yPredValid = []
    for p in PValid:
        yPredValid.append(np.argmax(p))

    print("VALIDAÇÃO ----> REGRESSÃO LOGISTICA ONE-VS-ALL - ALPHA", str(alpha), " - ", str(it), " ITERAÇÕES")
    print("F1 Score:" + str(f1_score(validY, yPredValid, average='micro')))

    # ============= TEST

    #Predictions
    P = sigmoid(testX.dot(all_theta.T)) # probability of each class

    yPred = []
    for p in P:
        yPred.append(np.argmax(p))

    """plt.scatter(results, range(len(results)), color='blue')
    plt.scatter(validY['label'].values, range(len(validY['label'].values)), color='red')
    plt.ylabel('Classe Predita')
    plt.xlabel('Observação')
    plt.title('Predição vs Real')
    plt.show()"""

    # Accuracy
    print("REGRESSÃO LOGISTICA ONE-VS-ALL - ALPHA", str(alpha), " - ", str(it), " ITERAÇÕES")
    print("F1 Score:" + str(f1_score(testY, yPred, average='micro')))

    # confusion matrix
    cm = confusion_matrix(testY, yPred)
    plt.figure()
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], normalize=False, title='Confusion Matrix')
    plt.show()