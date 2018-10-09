# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 02 - Multinomial Regression
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
def logisticRegressionMulticlass(X, y, w, it, alpha):
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

it = 1000
# tested alphas: 0.1, 0.01, 0.001 - Better: 0.1
alphas = [0.1]

for alpha in alphas:
    # execute multinomial regression
    J, w = logisticRegressionMulticlass(trainX, y, w, it, alpha)

    # plot graph for GD with regularization
    plt.plot(J)
    plt.ylabel('Função de custo J')
    plt.xlabel('Número de iterações')
    plt.title('Regressão Logistica Multinomial. Alpha = 0.1')
    plt.show()

    # ============== VALIDATION

    #Predictions
    PValid = softmax(validX.dot(w)) # probability of each class

    yPredValid = []
    for p in PValid:
        yPredValid.append(np.argmax(p))

    print("VALIDAÇÃO ----> REGRESSÃO LOGISTICA MULTINOMIAL - ALPHA", str(alpha), " - ", str(it), " ITERAÇÕES")
    print("F1 Score:" + str(f1_score(validY, yPredValid, average='micro')))

    # ============= TEST

    #Predictions
    P = softmax(testX.dot(w)) # probability of each class

    yPred = []
    for p in P:
        yPred.append(np.argmax(p))

    # Accuracy
    print("TESTE ----> REGRESSÃO LOGISTICA MULTINOMIAL - ALPHA", str(alpha), " - ", str(it), " ITERAÇÕES")
    print("F1 Score:" + str(f1_score(testY, yPred, average='micro')))

    # confusion matrix
    cm = confusion_matrix(testY, yPred)
    plt.figure()
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], normalize=False, title='Confusion Matrix')
    plt.show()
