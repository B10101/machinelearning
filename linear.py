import numpy as np
import matplotlib as plt
from sklearn.linear_model import LinearRegression

#SAmple Data
stamps = np.array([1,3,5,7,9]).reshape(-1,1)
amount = np.array([2,6,8,12,18])

model = LinearRegression()

model.fit(stamps,amount)

next = 10
predicted = model.predict([[next]])


print(predicted)
