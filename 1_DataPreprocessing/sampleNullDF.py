import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

# %% calling DF
dataFrame = pd.read_csv("../data/eksikveriler.csv")
print(dataFrame)

# %% filling null variables
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

nullValues = dataFrame.iloc[:, 1:4].values
print(nullValues)

nullValues = imputer.fit_transform(nullValues[:, 0:4])
print(nullValues)

# %% getting countries to encode
from sklearn import preprocessing

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

# %% merging all datas into one daa frame
resultDF1 = pd.DataFrame(data=country, index=range(22), columns=["fr", "tr", "usa"])
print(resultDF1)

resultDF2 = pd.DataFrame(data=nullValues, index=range(22), columns=["height", "weight", "age"])
print(resultDF2)

sex = dataFrame.iloc[:, -1].values
print(sex)
resultDF3 = pd.DataFrame(data=sex, index=range(22), columns=["sex"])
print(resultDF3)

# concatenation
s = pd.concat([resultDF1, resultDF2], axis=1)
print(s)

mergedDF = pd.concat([s, resultDF3], axis=1)
print(mergedDF)

# %% splitting mergedDF into 4 pieces(x_train, x_test, y_train, y_test)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, resultDF3, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
