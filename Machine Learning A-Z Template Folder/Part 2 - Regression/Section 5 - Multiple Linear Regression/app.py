# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/Menda0/Projects/machine-learning/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])

oneHotEnconder = OneHotEncoder(categorical_features=[3])
X = oneHotEnconder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
# X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
