import numpy as np 
import matplotlib.pyplot as plt


#To run the nereast neighbour classifier, we define a distance function
#to calculate distance between points
def dist(x,y):
	return np.sqrt(np.sum((x-y)**2))

X_train = np.array([[1,1],[2,2.5],[3,1.2],[5.5,6.3],[6,9],[7,6]])
Y_train = ['red','red','red','blue','blue','blue']

X_test = np.array([3,4])  #creating test point

# plotting

plt.figure()
#s = size
plt.scatter(X_train[:,0],X_train[:,1],s = 170 , color = Y_train[:])
#plotting the test point
plt.scatter(X_test[0],X_test[1],s=170,color='yellow')
plt.show()

num = len(X_train)
distance = np.zeros(num)
for i in range(num):
	distance[i] = dist(X_train[i],X_test)
print(distance)

min_index = np.argmin(distance) #index with smallest distacne
print(Y_train[min_index])
