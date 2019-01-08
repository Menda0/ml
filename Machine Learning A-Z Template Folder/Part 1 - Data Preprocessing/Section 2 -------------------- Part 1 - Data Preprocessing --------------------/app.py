import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data set
dataset = pd.read_csv("/Users/Menda0/Projects/machine-learning/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

oneHotEnconder = OneHotEncoder(categorical_features=[0])
X = oneHotEnconder.fit_transform(X).toarray()

labelEncoder_Y = LabelEncoder()
y = labelEncoder_X.fit_transform(y)

# Avoiding the dummy variable trap
X = X[:, 1:]

# from sklearn.model_selection import train_test_split
# Splitting the dataset into the training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

# Fitting multiple linerar regression to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction the Test set results
y_pred = regressor.predict(X_test)

# Buillding the optimalç model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
