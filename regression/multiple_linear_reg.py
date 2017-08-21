"""
Multiple Regression model in Python
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset - set up wd
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
  
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
   
# Avoid dummy variable trap
X = X[:, 1:]

# Split data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.2, random_state = 0)

# Feature scaling (-1, +1) range for values
"""from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

scale_Y = StandardScaler()
Y_train = scale_Y.fit_transform(Y_train)
"""

# Fitting Multiple Linear Regression to train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Backward elimination to build an optimal model
# - eliminate not statistically significant IVs
# compute p-values and eliminate variables that are not stat. significant
import statsmodels.formula.api as sm

# add column of ones at beginning of matrix of features
# - lib doesn't take into account b0 constant
# (b_0*x_0 part of formula)
X = np.append( arr = np.ones((50,1)).astype(int), values = X, axis = 1)

# only contains stat. significant independent variables
X_optimal = X[:, [0,1,2,3,4,5]] 
# create new model - ordinary least squares
regressor_ols = sm.OLS(endog = Y, exog = X_optimal).fit()
# examine statistical metrics to get p-values
# eliminate variables with p-value > 0.05
regressor_ols.summary()

X_optimal = X[:, [0,1,3,4,5]]   
regressor_ols = sm.OLS(endog = Y, exog = X_optimal).fit()
regressor_ols.summary()

X_optimal = X[:, [0,3,4,5]]   
regressor_ols = sm.OLS(endog = Y, exog = X_optimal).fit()
regressor_ols.summary()

X_optimal = X[:, [0,3,5]]   
regressor_ols = sm.OLS(endog = Y, exog = X_optimal).fit()
regressor_ols.summary()

X_optimal = X[:, [0,3]]   
regressor_ols = sm.OLS(endog = Y, exog = X_optimal).fit()
regressor_ols.summary()
