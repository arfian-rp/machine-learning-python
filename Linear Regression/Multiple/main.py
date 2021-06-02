import pandas as pd
import numpy as np
import sklearn.model_selection as sm
import sklearn.linear_model as lm

data = pd.read_csv('Kendaraan.csv')
X = data[['Liter', 'Penumpang', 'Suhu', 'Kecepatan']]
y = data[['Kilometer']]
X_train, X_test, y_train, y_test = sm.train_test_split(X, y, test_size=0.1, random_state=0)
model = lm.LinearRegression()
model.fit(X_train, y_train)
print('intercept = ', model.intercept_)
print('slope = ', model.coef_)
data = np.array([[30, 2, 10, 50]])# L, P, S, K
hasil = model.predict(data)
print(hasil)


