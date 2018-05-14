import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.cluster import KMeans

df = pd.read_csv('data.csv',delim_whitespace=True)
print(df.head())

#Assign th number of clusters
k = 2
kmeans = KMeans(n_clusters = k)

#Train the model
kmeans = kmeans.fit(df)

#array that contains cluster number
labels = kmeans.labels_

#array of size k with coordinates of centroid
centroids = kmeans.cluster_centers_

#Testing the model
x_test = [[4.671,67],[2.885,61],[1.666,90],[5.623,54],[2.678,80],[1.875,60]]

#Test the model(returns the cluster number)
prediction = kmeans.predict(x_test)
print(prediction)
