import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Fitting the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(6.5)

# Visualizing the data
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Decision Tree Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
