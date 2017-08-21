"""
Support Vector Regression in Python
"""
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values
 
'''
# Split data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                            test_size =0.2, random_state = 0)
'''
# Feature scaling (-1, +1) range for values
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
scale_Y = StandardScaler()
X= scale_X.fit_transform(X)
Y = scale_Y.fit_transform(Y)

# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
regressor = SVR( kernel = 'rbf')
regressor.fit(X, Y)

# Predicting new result with SVR Model
y_pred = scale_Y.inverse_transform(regressor.predict(scale_X.transform(np.array([[6.5]]) )))

# Visualizing the SVR Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Regression results 
# (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
 
 