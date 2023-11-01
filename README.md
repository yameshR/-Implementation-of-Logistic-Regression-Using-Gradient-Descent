# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### Step 1:
Use the standard libraries in python for finding linear regression.

#### Step 2:
Set variables for assigning dataset values.

#### Step 3:
Import linear regression from sklearn.

#### Step 4:
Predict the values of array.

#### Step 5:
Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

#### Step 6.
Obtain the graph.
## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: MARELLA DHARANESH
RegisterNumber:  212222240016
```
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt", delimiter = ",")
X = data[:, [0, 1]]
y = data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0],  X[y == 1][:, 1], label = "Admitted",color='red')
plt.scatter(X[y == 0][:, 0],  X[y == 0][:, 1], label = "Not Admitted",color='green')
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10, 10 , 100)
plt.plot(X_plot, sigmoid(X_plot),color='red')
plt.show()

def costFunction(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J, grad

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

def cost(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J

def gradient(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, h - y) / X.shape[0]
    return grad

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res = optimize.minimize(fun = cost, x0 = theta, args = (X_train, y), method = "Newton-CG", jac = gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted",color='red')
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Not admitted",color='green')
    plt.contour(xx, yy, y_plot, levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
plotDecisionBoundary(res.x, X, y)

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```
## Output:
#### Array value of X:

![Screenshot from 2023-10-07 09-06-01](https://github.com/Gchethankumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118348224/19a6306d-bc01-42a0-a7de-6eabab6b0c34)

#### Array value of Y:

![Screenshot from 2023-10-07 09-06-08](https://github.com/Gchethankumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118348224/e63b1338-7482-4ef8-bb8f-9dd4607e4043)


#### Exam 1- Score Graph:

![Screenshot from 2023-10-07 09-06-20](https://github.com/Gchethankumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118348224/f63d2b28-11b5-4e84-9f2c-bd6e50cb4a56)


#### Sigmoid function graph:

![Screenshot from 2023-10-07 09-06-27](https://github.com/Gchethankumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118348224/39ed6837-31d3-4558-a39e-b1e4f958481a)


#### X_Train grad value:

![Screenshot from 2023-10-07 09-06-41](https://github.com/Gchethankumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118348224/2ed3471e-3c1a-4e45-a948-fe6ff069798c)


#### Y_Train gradm value:

![Screenshot from 2023-10-07 09-06-49](https://github.com/Gchethankumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118348224/7b4015ec-4ac5-461e-b612-95d6d97fe954)


#### Print res of X:

![Screenshot from 2023-10-07 09-06-57](https://github.com/Gchethankumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118348224/c66094d1-0a3a-4f66-a7ad-581cbab9a527)


#### Decision Boundary- Graph for Exam Score:

![Screenshot from 2023-10-07 09-07-06](https://github.com/Gchethankumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118348224/1503ee01-7567-4f41-b9f8-a4dd69b50a95)


#### Probability value:

![Screenshot from 2023-10-07 09-07-15](https://github.com/Gchethankumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118348224/4c971759-c41d-4998-80c6-f9eff75c1c9b)


#### Prediction value of mean:

![Screenshot from 2023-10-07 09-07-22](https://github.com/Gchethankumar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118348224/eb30b702-246d-4d97-8d6f-4d4fddf1d96c)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
