# XG Boost in Python
# Created by Yanitsa M

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost  

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Part 1 - Data Preprocessing
 
# Encoding categorical data
# Encoding the independent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Split data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size = 0.2, random_state = 0)
# Fitting XGBoost on the training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, Y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

# Confusion matrix to evaluate performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

# Applying K-fold CV
from sklearn.model_selection import cross_val_score
# optional param n_jobs - faster computation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train,
                             cv = 10)
accuracies.mean()
accuracies.std()
