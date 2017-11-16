# Decision Tree Regression
# Very powerful model in more dimensions - not very interesting in 1D

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('../../data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # x is a matrix
y = dataset.iloc[:, 2].values # y is a vector

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Decision Tree Regression to the dataset
regressor = DecisionTreeRegressor(random_state = 0).fit(X, y)

# Visualizing the Decision Tree Regression results (higher resolution)
# model is non-linear and non-continuous
# the best way to visualize is in higher resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting result for x=6.5
y_pred = regressor.predict(6.5)