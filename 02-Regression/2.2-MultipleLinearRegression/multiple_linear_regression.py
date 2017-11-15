# Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

dataset = pd.read_csv('../../data/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# X has categorical variable. It needs to be encoded.

# Encoding categorical data
X[:, 3] = LabelEncoder().fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap:
# Removed the 1st column of X [0], so I dont need to worry about the dummy variable trap
# This is not really necessary because the library takes care of this,
# but it's good to know/remember
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set:
# 10 observations into test set (20%)
# 40 observations into training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#######################################################################
# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

# y = b0 + b1 x1 + b2 x2 +b3 x3 + bn xn
# I will add x0
# it will be 1, so it wont change the constant (b0)
# I have to add because the stats model does not have b0 = 1
# Most libraries have b0 included, such as sklearn

# (following above: Add X to a matrix of 1)
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# Create optimal matrix of features:
# it will contain only the variales that have high impact on the profit
# first add all (indexes) and remove step by step
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#significance level = 0.05
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print (regressor_OLS.summary())

# x2 has highest P value
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# remove x1
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#print (regressor_OLS.summary())

# remove x2 (index 4)
X_opt = X[:, [0, 3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#print (regressor_OLS.summary())

# Remove x2 (index 5)
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#print (regressor_OLS.summary())
