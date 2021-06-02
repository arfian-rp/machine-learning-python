import numpy as np
import matplotlib.pyplot as plt
usia = np.array([1,2,3,4,5,6,7])
ting = np.array([80,70,90,100,120,150,150,])
import sklearn.model_selection as ms
import vega_datasets as vd
data = vd.data.cars()
mpg = data[['Miles_per_Gallon']]
wil = data[['Weight_in_lbs']]
x_train, x_test, y_train, y_test = ms.train_test_split(wil, mpg, test_size=0.2, random_state=0)
import sklearn.linear_model as lm
mod = lm.LinearRegression()
mod.fit(x_train, y_train)
plt.scatter(usia, ting)
plt.show()