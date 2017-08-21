# Artificial Neural Network in Python
# Application: Demographic segmentation model for a bank (customer-centric orgs.)

# Numerical computations libraries
# Installing Theano 
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
# Feature scaling (-1, +1) range for values
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

# Part 2 - Build ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN 
classifier = Sequential()

# Add input layer,  first hidden layer, output layer
# output_dim = # hidden layer nodes
# input_dim = # input layer nodes
# activation function = rectifier for hidden, sigmoid for output
# (if more than 2 class labels - AF for output layer is softmax)
# initialization of weights = uniform distribution
classifier.add(Dense(activation= 'relu', input_dim= 11, units= 6, kernel_initializer= 'uniform')) 
classifier.add(Dense(activation= 'relu', units= 6, kernel_initializer= 'uniform')) 
classifier.add(Dense(activation= 'sigmoid', units= 1, kernel_initializer= 'uniform')) 

# Compiling the ANN 
# optimizer = find optimal set of weights (adam = stochastic gradient descent)
# loss = loss function; binary outcome = binary_crossentropy; > 2 categories: categorical_crossentropy
# metrics = evaluation criterion such as accuracy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN on the training data
classifier.fit(X_train, Y_train, batch_size = 10 , nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model
y_pred = classifier.predict(X_test)
# transform output from probabilities to True/False values
y_pred = y_pred > 0.5

# Confusion matrix to evaluate performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
# check accuracy (# correct/ total predictions)
 
