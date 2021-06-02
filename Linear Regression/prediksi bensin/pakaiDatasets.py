import numpy as np
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import vega_datasets as vd

data = vd.data.cars()
wil = data['Weight_in_lbs']
mpg = data['Miles_per_Gallon']
mpg = mpg.fillna(mpg.mean())
wil = np.array(wil)
mpg = np.array(mpg)
wil = wil.reshape(len(wil), 1)
mpg = mpg.reshape(len(mpg), 1)

print(np.min(wil), np.max(wil))

X_train, X_test, y_train, y_test = ms.train_test_split(wil, mpg)

model = lm.LinearRegression()
model.fit(X_train, y_train)
print(model.coef_, model.intercept_)

plt.scatter(X_train, y_train, edgecolors='r')
plt.xlabel('Weight_in_lbs')
plt.ylabel('Miles_per_Gallon')
plt.title('Linear Regression berat kendaraan dan bensin dalam 1 mil')
x1 = np.linspace(1613, 5140)
y1 = 46.490 + (-0.007) * x1
plt.plot(x1, y1)
milpergalon = model.predict([[7000]]) #berat
print(milpergalon)
plt.show()
