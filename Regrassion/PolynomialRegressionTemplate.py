# 1 - Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

#  2 - Data Preprocessing
# --> (2.1 - Reading data and calling DF)

dataFrame = pd.read_csv("Regrassion/maaslar.csv")
print(dataFrame.isnull().sum())

#  Splitting data
x = dataFrame.iloc[:, 1:2]
y = dataFrame.iloc[:, 2:]

# %% Creating linear regression model
from sklearn.linear_model import LinearRegression  # for linear regression

linearModel = LinearRegression()
linearModel.fit(x, y)

# Drawing graph
plt.scatter(x, y, color="r")
plt.plot(x, linearModel.predict(x), color="blue")
plt.show()

# %% Creating polynomial regression
from sklearn.preprocessing import PolynomialFeatures

polyModel = PolynomialFeatures(degree=2)
xPoly = polyModel.fit_transform(x.values)
print(xPoly)

linearModel2 = LinearRegression()
linearModel2.fit(xPoly, y)

plt.scatter(x, y, color="red")
plt.plot(x, linearModel2.predict(polyModel.fit_transform(x)))
plt.show()

polyModel = PolynomialFeatures(degree=4)
xPoly = polyModel.fit_transform(x.values)
print(xPoly)

linearModel2 = LinearRegression()
linearModel2.fit(xPoly, y)

plt.scatter(x, y, color="red")
plt.plot(x, linearModel2.predict(polyModel.fit_transform(x)))
plt.show()

# %% Guessing
print(linearModel.predict([[11]]))
print(linearModel.predict([[6.6]]))

print(linearModel2.predict(polyModel.fit_transform([[11]])))
print(linearModel2.predict(polyModel.fit_transform([[6.6]])))