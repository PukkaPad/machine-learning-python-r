# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

dataset = pd.read_csv('../../data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # x is a matrix
y = dataset.iloc[:, 2].values # y is a vector

plt.scatter(X, y, color = 'red')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Training set and Test set
"""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
# small number of observations (10)
# because of that dataset will not be split into training and test

# Fitting Linear Regression to the dataset
lin_reg = LinearRegression().fit(X, y)

# Fitting Polynomial Regression to the dataset
# transform matrix X into a X_poly
# which is a polinomial matrix of X
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
# create a new linear regression object
# to fit the polinomial
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue', label = 'Predicted (Linear Model)')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue', label = 'Predicted (Polinomial Model)')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))