Machine Learning - Python and R
-----------------------
Machine learning algorithms in Python and R

01 - Data Preprocessing
-----------------------

### Missing data

Handling missing data is important because a lot of ML algorithms do not support data with missing values. There are many approaches to missing values. For this example I have used **mean** to replace NaN. Mean imputation is a naive imputation method (unlike k-nearest neighbors for example).

By replacing missing with mean, the mean is preserved. The extremes will not change.

There are other approaches:

(replace NaN with)
* median
* interpolated estimate
* a constant
* dummy value
* 0
* delete rows
* etc

As a general rule, it's important to pay attention to why the data are missing. [More information](https://en.wikipedia.org/wiki/Missing_data).

### Categorical data

[Categorical variable](https://en.wikipedia.org/wiki/Categorical_variable)

### Splitting the dataset: training and test set
The model is fit on a training set. The test dataset is used to provide an unbiased evaluation of the model fit.


### Feature scaling

Data is standardized, so large variables will not dominate the computed dissimilarity


02 - Regression
-----------------------

Linear regression assumptions:

1. Linearity

2. [Homoscedasticity](https://en.wikipedia.org/wiki/Homoscedasticity):
variance around regression line is the same for all X

3. [Multivariate normality](https://en.wikipedia.org/wiki/Multivariate_normal_distribution):
joint distribution of a random vector that can be represented as a linear transformation

4. Independence of erros:
no correlation between consecutive errors

5. Lack of multicollinearity:
multicollinearity occurs when the independent variables are too highly correlated with each other

### 2.1 - Simple Linear Regression

### 2.2 - Multiple Linear Regression
Important to know:

[Dummy variable trap](http://www.algosome.com/articles/dummy-variable-trap-regression.html)

### 2.3 - Polynomial Regression

### 2.4 - Support Vector Regression (SVR)

[SVR](http://www.saedsayad.com/support_vector_machine_reg.htm)

### 2.5 - Decision Tree Regression

### 2.6 - Random Forest Regression

### 2.7 - Evaluating Regression Models Performance

* R-Squared: tells how good the regression line is compared to the average line of the dataset; it tells how well the model is fitted to the data.

* Adjusted R-Squared: R-Squared will always increase. So if I add variables to the model, I will not know if the variables are helping the model or not.
The adjusted R-Squared is then used instead


03 - Classification
-----------------------

### 3.1 - Logistic Regression

### 3.2 - K-Nearest Neighbors
Idea: search for closest match of the test data in the feature space

### 3.3 - Support Vector Machine

### 3.4 - Kernel Support Verctor Machine
Kernel trick: implicitly work in a higher-dimensional space, without explicitly building the higher-dimensional representation

Types of kernel functions:
[info](http://mlkernels.readthedocs.io/en/latest/kernels.html)
* RBF (Gaussian)
* Sigmoid
* Polynomial

### 3.5 - Naive Bayes
* Bayes' Theorem
It's called Naive because Bayes Theorem requires some **Independence assumption**, these assumptions are often times not correct. It requires that the variables are independent.


04 - Clustering
-----------------------

05 - Association Rule Learning
-----------------------

06 - Reinforcement Learning
-----------------------

7 - Natural Language Processing
-----------------------


8 - Deep Learning
-----------------------


9 - Dimensionality Reduction
-----------------------


10 - Model Selection & Boosting
-----------------------

