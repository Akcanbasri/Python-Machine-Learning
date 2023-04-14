# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

dataFrame = pd.read_csv("data/veriler.csv")
print(dataFrame)

height = dataFrame[["boy"]]
plt.show()
print(height)

dataFrame.plot()
plt.show()

