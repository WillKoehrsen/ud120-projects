from __future__ import print_function
from sklearn import linear_model
import numpy as np

X = np.array([[500,0],[500,20],[500,40],[1000,0],[1000,20],[1000,40],[1500,0],[1500,20], [1500,40]])
y = np.array([1000, 800, 600, 1500, 1300, 1100, 2000, 1800, 1600])

reg = linear_model.LinearRegression()
reg.fit(X,y)
print(reg.coef_)
print(reg.intercept_)