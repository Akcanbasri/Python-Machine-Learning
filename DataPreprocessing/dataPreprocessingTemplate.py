# 1 - Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.impute import SimpleImputer  # for filling null variables
from sklearn import preprocessing  # encoding for categorical variables
from sklearn.model_selection import train_test_split  # for splitting data
from sklearn.preprocessing import StandardScaler  # for standardization of data

# 2 - Data Preprocessing
# %%--> (2.1 - Reading data and calling DF)
dataFrame = pd.read_csv("../data/eksikveriler.csv")

# %%--> (2.2 - Filling null variables)
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

nullValues = dataFrame.iloc[:, 1:4].values
# DF before filling
print(nullValues)

nullValues = imputer.fit_transform(nullValues[:, 0:4])
# DF after filling
print(nullValues)

# %%--> (2.3 - Encoder Nominal,Ordinal -> Numeric)
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

# %%--> (2.4 - Converting np series into pd Data Frame)
resultDF1 = pd.DataFrame(data=country, index=range(22), columns=["fr", "tr", "usa"])
print(resultDF1)

resultDF2 = pd.DataFrame(data=nullValues, index=range(22), columns=["height", "weight", "age"])
print(resultDF2)

sex = dataFrame.iloc[:, -1].values
print(sex)
resultDF3 = pd.DataFrame(data=sex, index=range(22), columns=["sex"])
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
