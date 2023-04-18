# 1 - Libraries
import matplotlib.pyplot as plt
import pandas as pd

#  2 - Data Preprocessing
# --> (2.1 - Reading data and calling DF)

dataFrame = pd.read_csv("Regrassion/maaslar.csv")
print(dataFrame.isnull().sum())

#  Splitting data
x = dataFrame.iloc[:, 1:2]
y = dataFrame.iloc[:, 2:]

X = x.values
Y = y.values

#  3 - Modelling
# %% Creating linear regression model
from sklearn.linear_model import LinearRegression  # for linear regression

linearModel = LinearRegression()
linearModel.fit(X, Y)

# %% Creating polynomial regression (nonlinear regression)
from sklearn.preprocessing import PolynomialFeatures
# 2th degree polynomial regression
polyModel = PolynomialFeatures(degree=2)
xPoly = polyModel.fit_transform(x.values)
linearModel2 = LinearRegression()
linearModel2.fit(xPoly, y)

# 4th degree polynomial regression
polyModel3 = PolynomialFeatures(degree=4)
xPoly = polyModel3.fit_transform(x.values)
linearModel3 = LinearRegression()
linearModel3.fit(xPoly, y)

# %% Drawing graph
#  for linear regression
plt.scatter(X, Y, color="r")
plt.plot(x, linearModel.predict(X), color="blue")
plt.show()
# graph for 2nd degree polynomial regression
plt.scatter(x, y, color="red")
plt.plot(x, linearModel2.predict(polyModel.fit_transform(x)))
plt.show()
# graph for 4th degree polynomial regression
plt.scatter(x, y, color="red")
plt.plot(x, linearModel3.predict(polyModel3.fit_transform(x)))
plt.show()

# %% Guessing
print(linearModel.predict([[11]]))
print(linearModel.predict([[6.6]]))

print(linearModel2.predict(polyModel.fit_transform([[11]])))
print(linearModel2.predict(polyModel.fit_transform([[6.6]])))
