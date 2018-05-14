# Polynomial Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# take all the rows and take all the columns except the last column
# the last column in a dependent variable column
X = dataset.iloc[:,1:2].values     # to specify that this is a matrix of features and not a vector

# selecting the last column
y = dataset.iloc[:,2].values

# no need to divide the dataset into training set and testing set

# no need to feature scaling either

# Fitting the dataset into Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(X,y)

# Fitting the dataset into Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) # at first it was 2
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# visualizing the Linear Regression Model
plt.scatter(X,y, color='red')
plt.plot(X, lin_reg1.predict(X), color='blue')
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# visualizing the Polynomial Model
plt.scatter(X,y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# for better resolution
X_grid = np.arange(min(X), max(X), 0.01) # the last value improves the precision by that value
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# predicting the values
# using Linear Model
lin_reg1.predict(6.5)   # we cannot plot it against another matrix ot vector

#Using polynomial model
lin_reg2.predict(poly_reg.fit_transform(6.5))
