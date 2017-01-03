# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 23:39:02 2017

@author: lancel
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('datasets/ann_train/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising ANN model
ann_classifier = Sequential()
ann_classifier.add(Dense(output_dim = 6, input_dim = 11, init = 'uniform', activation = 'relu'))# input layer
ann_classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) # second hidden layer
ann_classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) # output layer

# Compiling and fitting ANN
ann_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann_classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting test
y_pred = ann_classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# evaluations
(loss, accuracy) = ann_classifier.evaluate(X_test, y_test, batch_size = 10, verbose = 1)
print("[INFO] loss = {:.4f}, accuracy = {:.4f}%".format(loss, accuracy * 100))

ann_cm = confusion_matrix(y_test, y_pred)