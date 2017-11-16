# Random Forest Regression
# It's a version of Ensemble Learning
# Ensemble Learning is when you take multiple algorithms or the same algorithms multiple times and they are put together to create something more powerful than the original

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('../../data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # x is a matrix
y = dataset.iloc[:, 2].values # y is a vector

# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting result for 6.5
y_pred = regressor.predict(6.5)

print('Salary prediction is {0}'.format(y_pred))