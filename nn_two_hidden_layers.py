# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 02 - Neural Network w/ One Hidden Layer
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sb

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

def leaky_relu_derivative(z):
    def do_leaky_deriv(x):
        if x > 0:
            return 1
        else:
            return 0.01

    relufunc = np.vectorize(do_leaky_deriv)
    return relufunc(z)

def leaky_relu(z):
    maxValue = z.max()
    def do_leaky(x):
        if x > 0:
            return x / maxValue
        else:
            return 0.01 * x / maxValue
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
        a1 = leaky_relu(z1)

        # Second activation function
        # z2 = f(W2 * z1 + b2)
        z2 = np.dot(a1, weights2)
        a2 = leaky_relu(z2)

        # Third activation function(softmax)
        # z3 = softmax(W3 * z2 + b3)
        z3 = np.dot(a2, weights3)
        a3 = softmax(z3)

        # step 3 - calculate ouput error
        J.append(loss(a3, y))

        # step 4 - calculate derivative of error
        # step 5 - backpropagate
        dz3 = (y - a3)
        dw3 = a2.T.dot(dz3)

        dz2 = dz3.dot(weights3.T) * leaky_relu_derivative(a2)
        dw2 = a1.T.dot(dz2)

        dz1 = dz2.dot(weights2.T) * leaky_relu_derivative(a1)
        dw1 = x.T.dot(dz1)

        # step 6 - update the weights with the derivative cost function
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
it = 1000
alphas = [0.0001, 0.00001, 0.000001]

for alpha in alphas:
    weights, J = NeuralNetwork(trainX, trainY, it, alpha)

    # plot graph for GD
    plt.plot(J)
    plt.ylabel('Função de custo J')
    plt.xlabel('Número de iterações')
    plt.title('Rede Neural com duas camadas escondidas')
    plt.show()

    # ============== VALIDATION

    # predict value with validation
    z1Valid = leaky_relu(np.dot(validX, weights[0]))
    z2Valid = leaky_relu(np.dot(z1Valid, weights[1]))
    resultsValid = softmax(np.dot(z2Valid, weights[2]))

    yPredValid = []
    for r in resultsValid:
        yPredValid.append(np.argmax(r))

    # Accuracy
    print("VALIDAÇÃO ----> REDE NEURAL COM 2 CAMADAS ESCONDIDAS - ALPHA", str(alpha), " - ", str(it), " ITERAÇÕES")
    print("F1 Score:" + str(f1_score(validY, yPredValid, average='micro')))

    # ============= TEST

    # predict value with validation
    z1 = leaky_relu(np.dot(testX, weights[0]))
    z2 = leaky_relu(np.dot(z1, weights[1]))
    results = softmax(np.dot(z2, weights[2]))

    # get highest probability for each observation
    yPred = []
    for r in results:
        yPred.append(np.argmax(r))

    # Accuracy
    print("TESTE ----> REDE NEURAL COM 2 CAMADAS ESCONDIDAS - ALPHA", str(alpha), " - ", str(it), " ITERAÇÕES")
    print("F1 Score:" + str(f1_score(testY, yPred, average='micro')))

    # confusion matrix
    cm = confusion_matrix(testY, yPred)
    plt.figure()
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], normalize=False, title='Confusion Matrix')
    plt.show()
