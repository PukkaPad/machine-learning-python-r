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