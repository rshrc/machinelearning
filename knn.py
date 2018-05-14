import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data.csv',delim_whitespace=True)
print(df.head())

#Preparing the training set
x = df.loc[:,'sepal_length':'petal_width']
y = df.loc[:,'species']

#training the model
knn = KNeighborsClassifier()
knn.fit(x,y)

x_test = [[4.9,7.0,1.2,0.2],[6.0,2.9,4.5,1.5],[6.1,2.6,5.6,1.2]]

prediction = knn.predict(x_test)

print(prediction)
