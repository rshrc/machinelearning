import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# take all the rows and take all the columns except the last column
# the last column in a dependent variable column
X = dataset.iloc[:,1:2].values

# selecting the last column
y = dataset.iloc[:,2].values
y = y.reshape(-1,1)
# Feature Scaling is required
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting the dataset into the SVR
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Truth or Bluff(SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

