import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

my_data = pd.read_csv('home.txt',names=["size","bedroom","price"])

my_data = (my_data - my_data.mean())/my_data.std()
my_data.head()

X = my_data.iloc[:,0:2]

ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
#print(X)
#print(len(X))

y = my_data.iloc[:,2:3]

theta = np.zeros([1,3])

alpha = 0.01
iters = 1000
#print(y)

def computeCost(X,y,theta):
    inner = np.power(((X @ theta.T)-y),2)
    return np.sum(inner)/(2 *len(X))

print(computeCost(X,y,theta))

def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        h = X @ theta.T
        temp = h-y
        S = X.T @ temp
        S = (alpha/len(X))*S
        theta = theta - S.T
        c = computeCost(X,y,theta)
        cost[i] = c
        
    return cost,theta

g,thita = gradientDescent(X,y,theta,iters,alpha)

c = computeCost(X,y,thita)
print("Final cost ",c)

fig, ax = plt.subplots()  
ax.plot(np.arange(iters), g, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch') 