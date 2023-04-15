# 1 - Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.impute import SimpleImputer  # for filling null variables
from sklearn import preprocessing  # encoding for categorical variables
from sklearn.model_selection import train_test_split  # for splitting data
from sklearn.preprocessing import StandardScaler  # for standardization of data
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm  # getting statics values info about our model

# 2 - Data Preprocessing
#   --> (2.1 - Reading data and calling DF)
dataFrame = pd.read_csv("../data/veriler.csv")
print(dataFrame.isnull().sum())

# %%--> (2.2 - Filling null variables)
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
nullValues = dataFrame.iloc[:, 1:4].values
# DF before filling
print(nullValues)
nullValues = imputer.fit_transform(nullValues[:, 0:4])
# DF after filling
print(nullValues)

# %%--> (2.3 - Encoder Nominal,Ordinal -> Numeric)
# encoder for country
country = dataFrame.iloc[:, 0:1].values
print(country)
# label encoder
le = preprocessing.LabelEncoder()
country[:, 0] = le.fit_transform(dataFrame.iloc[:, 0])
print(country)
# ohe = one hot encoder
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)

# encoder for sexuality
sex = dataFrame.iloc[:, -1:].values
print(sex)
# label encoder
le = preprocessing.LabelEncoder()
sex[:, -1] = le.fit_transform(dataFrame.iloc[:, -1])
print(sex)
# ohe = one hot encoder
ohe = preprocessing.OneHotEncoder()
sex = ohe.fit_transform(sex).toarray()
print(sex)

# %%--> (2.4 - Converting np series into pd Data Frame)
resultDF1 = pd.DataFrame(data=country, index=range(22), columns=["fr", "tr", "usa"])
print(resultDF1)

resultDF2 = pd.DataFrame(data=nullValues, index=range(22), columns=["height", "weight", "age"])
print(resultDF2)

resultDF3 = pd.DataFrame(data=sex[:, :1], index=range(22), columns=["sex"])
print(resultDF3)

# %%--> (2.5 - Concatenation of DataFrames)
s = pd.concat([resultDF1, resultDF2], axis=1)
print(s)

mergedDF = pd.concat([s, resultDF3], axis=1)
print(mergedDF)

# %%--> (2.6 - Splitting DF into 4 pieces(x_train, x_test,y_train,y_test) for learning)
x_train, x_test, y_train, y_test = train_test_split(s, resultDF3, test_size=0.33, random_state=0)

# %%--> (2.7 - Standardization of values for learning )
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# %% Creating model
model = LinearRegression()
model.fit(x_train, y_train)

yPred = model.predict(x_test)

# %% model for height
height = mergedDF[["height"]].values
x = mergedDF.drop("height", axis=1).values
xDF = pd.DataFrame(data=x, columns=["fr", "tr", "usa", "weight", "age", "sex"])
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, height, test_size=0.33, random_state=0)

model2 = LinearRegression()
model2.fit(x_train2, y_train2)

yPred2 = model2.predict(x_test2)

# %% Backward elimination
X = np.append(arr=np.ones((22, 1)).astype(int), values=xDF, axis=1)

XList = xDF.iloc[:, [0, 1, 2, 3, 4, 5]].values
XList = np.array(XList, dtype=float)
model3 = sm.OLS(height, XList).fit()
print(model3.summary())

XList = xDF.iloc[:, [0, 1, 2, 3, 5]].values
XList = np.array(XList, dtype=float)
model3 = sm.OLS(height, XList).fit()
print(model3.summary())

XList = xDF.iloc[:, [0, 1, 2, 3]].values
XList = np.array(XList, dtype=float)
model3 = sm.OLS(height, XList).fit()
print(model3.summary())
