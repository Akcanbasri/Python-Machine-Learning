# 1 - Libraries
import matplotlib.pyplot as plt
import pandas as pd

#  2 - Data Preprocessing
# --> (2.1 - Reading data and calling DF)

dataFrame = pd.read_csv("Prediction/maaslar.csv")
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

# %%--> (2.7 - Standardization of values for learning )
from sklearn.preprocessing import StandardScaler  # for standardization of data

sc1 = StandardScaler()
xScaled = sc1.fit_transform(X)
sc2 = StandardScaler()
yScaled = sc2.fit_transform(Y)

# %% Creating Support Vector Regression Model
from sklearn.svm import SVR

svrModel = SVR(kernel="rbf")
svrModel.fit(xScaled, yScaled.ravel())

# %% Drawing graph
plt.scatter(xScaled, yScaled, color="red")
plt.plot(xScaled, svrModel.predict(xScaled), color="blue")
plt.show()

# %% Guessing
print(svrModel.predict(sc1.transform([[11]])))
print(svrModel.predict(sc1.transform([[6.6]])))

# %% Creating decision tree model
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)

plt.scatter(X, Y, color="red")
plt.plot(X, r_dt.predict(X), color="blue")
plt.show()

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))