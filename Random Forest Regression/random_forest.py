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


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)
# n_estimators are the number of trees. More the tress, better the predictions
y_pred = regressor.predict(6.5)

# for better resolution
X_grid = np.arange(min(X), max(X), 0.01) # the last value improves the precision by that value
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

