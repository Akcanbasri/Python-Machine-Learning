# 1 - Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

# 2 - Data Preprocessing
# %%--> (2.1 - Reading data and calling DF)
dataFrame = pd.read_csv("3_Classification/veriler.csv")


# %%--> (2.3 - Encoder of Nominal,Ordinal -> Numeric)
from sklearn import preprocessing  # encoding for categorical variables

encodedDF = dataFrame.apply(preprocessing.LabelEncoder().fit_transform)

country = encodedDF.iloc[:, 0:1]
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)

# %%--> (2.4 - Converting np series into pd Data Frame)
countryDf = pd.DataFrame(data=country, index=range(22), columns=["fr", "tr", "usa"])
print(countryDf)

sex = encodedDF.iloc[:, -1].values
print(sex)
sexDF = pd.DataFrame(data=sex, index=range(22), columns=["sex"])
print(sexDF)

# %%--> (2.5 - Concatenation of DataFrames)
s = pd.concat([countryDf, dataFrame.iloc[:, 1:4]], axis=1)
print(s)

mergedDF = pd.concat([s, sexDF], axis=1)
print(mergedDF)

# %%--> (2.6 - Splitting DF into 4 pieces(x_train, x_test,y_train,y_test) for learning)
from sklearn.model_selection import train_test_split  # for splitting data

x_train, x_test, y_train, y_test = train_test_split(s, sexDF, test_size=0.33, random_state=0)

# %%--> (2.7 - Standardization of values for learning )
from sklearn.preprocessing import StandardScaler  # for standardization of data

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# %%--> (2.8 - Logistic Regression)
from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression(random_state=0)
logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)
print(y_pred)
print(y_test)

# %%--> (2.9 - Confusion Matrix)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)