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

# 2.1 - Simple Linear Regression


03 - Classification
-----------------------


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

