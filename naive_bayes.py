import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import GaussianNB

#importing dataset
df = pd.read_csv('data.csv',delim_whitespace=True)

print(df.head())

#preparing data for training
x = df.loc[:,'Age':'Nodes']
y = df.loc[:,'Survived']

#train the model
clf = GaussianNB()

clf.fit(x,y)

prediction = clf.predict([[12,70,12],[13,20,13]])
print(prediction)
