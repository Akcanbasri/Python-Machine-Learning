# 1 - Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import r2_score
import statsmodels.api as sm

# 2 - Data Preprocessing
# --> (2.1 - Reading data and calling DF)
dataFrame = pd.read_csv("Homeworks/maaslar_yeni.csv")

# %%--> (2.6 - Splitting DF into 4 pieces(x_train, x_test,y_train,y_test) for learning)
from sklearn.model_selection import train_test_split  # for splitting data

x = dataFrame.iloc[:, 2:5]
y = dataFrame.iloc[:, 5:]
X = x.values
Y = y.values

print(dataFrame.corr(numeric_only=True))
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# %% Creating linear regression model
from sklearn.linear_model import LinearRegression  # for linear regression

linearModel = LinearRegression()
linearModel.fit(X, Y)

# Backward elimination
print("Linear OLS")
model = sm.OLS(linearModel.predict(X), X).fit()
print(model.summary())

# r2 score for linear regression
print("Linear regression r2 score")
print(r2_score(Y, linearModel.predict(X)))

# %% Creating polynomial regression (nonlinear regression)
from sklearn.preprocessing import PolynomialFeatures

# 4th degree polynomial regression
polyModel = PolynomialFeatures(degree=4)
xPoly = polyModel.fit_transform(x.values)
print(xPoly)

#  Creating linear regression model
poLinearModel = LinearRegression()
poLinearModel.fit(xPoly, y)

# Backward elimination
print("Polynomial OLS")
model2 = sm.OLS(poLinearModel.predict(polyModel.fit_transform(x)), x).fit()
print(model2.summary())

#  R2 Score for polynomial regression
print("Polinomial regression r2 score")
print(r2_score(y, poLinearModel.predict(polyModel.fit_transform(x))))

# %%--> (2.7 - Standardization of values for learning )
#  Creating Support Vector Regression Model
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler  # for standardization of data

sc1 = StandardScaler()
xScaled = sc1.fit_transform(X)
sc2 = StandardScaler()
yScaled = sc2.fit_transform(Y)

svrModel = SVR(kernel="rbf")
svrModel.fit(xScaled, yScaled.ravel())

# Backward elimination
print("SVR OLS")
model3 = sm.OLS(svrModel.predict(xScaled), xScaled).fit()
print(model3.summary())

#  R2 Score for SVR
print("Support Vector Regression r2 score")
print(r2_score(yScaled, svrModel.predict(xScaled)))

# %% Creating decision tree model
from sklearn.tree import DecisionTreeRegressor

dtModel = DecisionTreeRegressor(random_state=0)
dtModel.fit(X, Y)


# Backward elimination
print("Decision Tree OLS")
model4 = sm.OLS(dtModel.predict(X), X).fit()
print(model4.summary())

#  R2 Score for DT
print("Decision Three r2 score")
print(r2_score(Y, dtModel.predict(X)))

# %% Creating random forest model
from sklearn.ensemble import RandomForestRegressor

rfModel = RandomForestRegressor(n_estimators=10, random_state=0)
rfModel.fit(X, Y.ravel())

# backward elimination
print("Random Forest OLS")
model5 = sm.OLS(rfModel.predict(X), X).fit()
print(model5.summary())

#  R2 Score for RFR
print("random forest r2 score")
print(r2_score(Y, rfModel.predict(X)))

# %% Summarize of R2 Score
print("Linear regression r2 score")
print(r2_score(Y, linearModel.predict(X)))
print("-------------------------------------------------------------------------------------")
print("Polinomial regression r2 score")
print(r2_score(y, poLinearModel.predict(polyModel.fit_transform(x))))
print("-------------------------------------------------------------------------------------")
print("Support Vector Regression r2 score")
print(r2_score(yScaled, svrModel.predict(xScaled)))
print("-------------------------------------------------------------------------------------")
print("Decision Three r2 score")
print(r2_score(Y, dtModel.predict(X)))
print("-------------------------------------------------------------------------------------")
print("random forest r2 score")
print(r2_score(Y, rfModel.predict(X)))
