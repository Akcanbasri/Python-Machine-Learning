# 1 - Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

# %% 2 - Data Preprocessing
#   --> (2.1 - Reading data and calling DF)
dataFrame = pd.read_csv("odev_tenis.csv")

# %%--> (2.2 - Filling null variables)
from sklearn.impute import SimpleImputer  # for filling null variables

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

nullValues = dataFrame.iloc[:, 1:3].values
# DF before filling
print(nullValues)

nullValues = imputer.fit_transform(nullValues[:, 0:3])
# DF after filling
print(nullValues)

# %%--> (2.3 - Encoder Nominal,Ordinal -> Numeric)
from sklearn import preprocessing  # encoding for categorical variables

encodedDF = dataFrame.apply(preprocessing.LabelEncoder().fit_transform)

outlook = encodedDF.iloc[:, :1]
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)

# %%--> (2.4 - Converting np series into pd Data Frame)
outlookDF = pd.DataFrame(data=outlook, index=range(14), columns=["overcast", "rainy", "sunny"])
print(outlookDF)

numericDF = pd.DataFrame(data=nullValues, index=range(14), columns=["temperature", "humidity"])
print(numericDF)

windyPlayDF = pd.DataFrame(data=encodedDF.iloc[:, -2:], index=range(14), columns=["windy", "play"])
print(windyPlayDF)

# %%--> (2.5 - Concatenation of DataFrames)
s = pd.concat([windyPlayDF, outlookDF], axis=1)
print(s)

mergedDF = pd.concat([s, numericDF], axis=1)
print(mergedDF)

# %%--> (2.6 - Splitting DF into 4 pieces(x_train, x_test,y_train,y_test) for learning)
from sklearn.model_selection import train_test_split  # for splitting data

x_train, x_test, y_train, y_test = train_test_split(mergedDF.iloc[:, :-1], mergedDF.iloc[:, -1:], test_size=0.33,
                                                    random_state=0)

# %%--> (2.7 - Standardization of values for learning )
from sklearn.preprocessing import StandardScaler  # for standardization of data

"""sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
"""
# %% Creating Linear Regression model
from sklearn.linear_model import LinearRegression  # for linear regression

model = LinearRegression()
model.fit(x_train, y_train)
predict = model.predict(x_test)
print(predict)

# %% Backward elimination
import statsmodels.api as sm  # getting statics values info about our model

X = np.append(arr=np.ones((14, 1)).astype(int), values=mergedDF.iloc[:, :-1], axis=1)
XList = mergedDF.iloc[:, [0, 1, 2, 3, 4, 5]].values
XList = np.array(XList, dtype=float)
backwardModel = sm.OLS(mergedDF.iloc[:, -1:], XList).fit()
print(backwardModel.summary())

# %% eliminating of highest P value
mergedDF = mergedDF.iloc[:, 1:]
XList = mergedDF.iloc[:, [0, 1, 2, 3, 4]].values
XList = np.array(XList, dtype=float)
backwardModel = sm.OLS(mergedDF.iloc[:, -1:], XList).fit()
print(backwardModel.summary())

# %% Recreating model after elimination
x_train2 = x_train.iloc[:, 1:]
x_test2 = x_test.iloc[:, 1:]

model.fit(x_train2, y_train)
predict = model.predict(x_test2)
print(predict)
