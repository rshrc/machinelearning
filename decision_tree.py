import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('data.csv',delim_whitespace=True)
print(df.head())

#preparing data for training
x_train = df.loc[:,'buying':'safety']
y_train = df.loc[:,'values']

#training the model
tree = DecisionTreeClassifier(max_leaf_nodes=3,random_state=0)

#train the model
tree.fit(x_train,y_train)

#testing the model
prediction = tree.predict([[4,3,2,1,2,3]])
print(prediction)
