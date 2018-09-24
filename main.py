# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 02
# Tamara Martinelli de Campos - RA 157324
# João Vítor B. Silva - RA 155951

import sys
import numpy as np
import pandas as pd

## MAIN

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

# normalize features
for c in trainX.columns.values:
    if(c != 'bias'):
        max = float(trainX[c].max())
        min = float(trainX[c].min())

        if (max == min):
            max = min + 1

        diff = float(max - min)

        trainX[c] -= min
        trainX[c]  /= diff

        validX[c] -= min
        validX[c] /= diff

# calculate cost
m, n = trainX.shape
r = 10
it = 100000
alpha = 0.1
