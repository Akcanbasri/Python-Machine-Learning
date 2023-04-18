import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

# %% Calling DF
dataFrame = pd.read_csv("../data/satislar.csv")
# checking null values
print(dataFrame.isnull().sum())

# %% Separating Datas
x = dataFrame[["Aylar"]]
y = dataFrame[["Satislar"]]

# %% Splitting Data
from sklearn.model_selection import train_test_split  # for splitting data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# %% Standardization
from sklearn.preprocessing import StandardScaler  # for standardization of data

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# %% Creating model(Linear Regression)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predict = model.predict(X_test)

# %% Drawing graph
X_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(X_train, y_train, "r")
plt.plot(x_test, predict, color="#000000")
plt.xlabel("month")
plt.ylabel("sales")
plt.title("sales chart by month")
plt.show()

# %% Testing model
# Make predictions on test data
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared score
r2score = r2_score(y_test, y_pred)

# Print the results
print("Mean Squared Error (MSE):", mse)
print("R-squared score:", r2score)

# %% Visualize the predicted results compared to the actual values
plt.scatter(x_test, y_test, color="red", label="Actual")
plt.plot(x_test, y_pred, color="black", label="Predicted")
plt.xlabel("month")
plt.ylabel("sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.show()
