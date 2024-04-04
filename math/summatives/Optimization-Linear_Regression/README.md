# Optimization Using Gradient Descent: Linear Regression

In this assignment, you will build a simple linear regression model to predict sales based on TV marketing expenses. You will investigate three different approaches to this problem: using `NumPy` and `Scikit-Learn` linear regression models, as well as constructing and optimizing the sum of squares cost function with gradient descent from scratch.

## Table of Contents

- [1 - Open the Dataset and State the Problem](#1---open-the-dataset-and-state-the-problem)
  - [Exercise 1](#exercise-1)
- [2 - Linear Regression in Python with `NumPy` and `Scikit-Learn`](#2---linear-regression-in-python-with-numpy-and-scikit-learn)
  - [2.1 - Linear Regression with `NumPy`](#21---linear-regression-with-numpy)
    - [Exercise 2](#exercise-2)
  - [2.2 - Linear Regression with `Scikit-Learn`](#22---linear-regression-with-scikit-learn)
    - [Exercise 3](#exercise-3)
    - [Exercise 4](#exercise-4)
- [3 - Linear Regression using Gradient Descent](#3---linear-regression-using-gradient-descent)
  - [Exercise 5](#exercise-5)
  - [Exercise 6](#exercise-6)
- [FastAPI Integration](#fastapi-integration)

## Packages

Load the required packages:

```python
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
```

1 - Open the Dataset and State the Problem
In this lab, you will build a linear regression model for a simple dataset, saved in a file data/tvmarketing.csv. The dataset has only two fields: TV marketing expenses (TV) and sales amount (Sales).

Exercise 1
Use the pandas function pd.read_csv to open the .csv file from the path.
 
```
path = "data/tvmarketing.csv"
adv = pd.read_csv(path)
print("Head of the training set is: ", adv.head())
print()
print("Information about the training set: ", adv.info())
print()
print("Shape of the training set: ", adv.shape)
```
2 - Linear Regression in Python with NumPy and Scikit-Learn

2.1 - Linear Regression with NumPy
You can use the function np.polyfit(x, y, deg) to fit a polynomial of degree deg to points (x, y), minimizing the sum of squared errors.

Exercise 2
Make predictions using the obtained slope and intercept coefficients.

```
X = np.array(adv['TV'])
Y = np.array(adv['Sales'])
m_numpy, b_numpy = np.polyfit(X, Y, 1)
X_pred = np.array([50, 120, 280])
Y_pred_numpy = m_numpy * X_pred + b_numpy
```

2.2 - Linear Regression with Scikit-Learn
Create an estimator object for a linear regression model and fit it to the training data.

Exercise 3
Fit the linear regression model using Scikit-Learn.

```
lr_sklearn = LinearRegression()
X_sklearn = X[:, np.newaxis]
Y_sklearn = Y[:, np.newaxis]
X_train, X_test, Y_train, Y_test = train_test_split(X_sklearn, Y_sklearn, test_size=0.2, random_state=0)
lr_sklearn.fit(X_train, Y_train)
```
Exercise 4
Make predictions using the fitted model and calculate the RMSE.

```
Y_pred = lr_sklearn.predict(X_test)
mse = sk.metrics.mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
```
3 - Linear Regression using Gradient Descent
Implement gradient descent to find linear regression coefficients.

Exercise 5
Define functions to calculate partial derivatives.

```
def dEdm(m, b, X, Y):
    ...

def dEdb(m, b, X, Y):
    ...
```
Exercise 6
Implement gradient descent using the calculated derivatives.

```
def gradient_descent(dEdm, dEdb, m, b, X, Y, learning_rate=0.001, num_iterations=1000, print_cost=False):
    ...

m_initial = 0
b_initial = 0
num_iterations = 30
learning_rate = 1.2
m_gd, b_gd = gradient_descent(dEdm, dEdb, m_initial, b_initial, X_norm, Y_norm, learning_rate, num_iterations, print_cost=True)
```

Fast API endpoint:
To integrate the linear regression model with FastAPI for predictions, follow the provided code snippet:

Here is a [link to my Swagger UI](https://www.example.com/swagger).

Here is a [link to my Video demonstrating how my API is working()]
