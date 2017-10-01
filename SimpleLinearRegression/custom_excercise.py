# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:54:22 2017

@author: Chemical
"""
#Simple linear regression 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('esercizio_custom.csv',delimiter = ';')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


#fitting linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
print (regressor.score(X_train,y_train))

# predicting the test set results
y_pred = regressor.predict(X_test)

#visualizing training set result
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Future selling (Training set)')
plt.xlabel('Customer Id')
plt.ylabel('Weekly Sales')
plt.show()

#visualizing test set result
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue') #punto di vista train
plt.title('Future selling (Test set)')
plt.xlabel('Customer Id')
plt.ylabel('Weekly Sales')
plt.show()