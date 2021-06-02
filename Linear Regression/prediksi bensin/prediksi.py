import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

dict1 = {
    'L':[20,25,20,30,40,35,30,30,25,25,25],
    'KM':[142,177,144,203,273,239,201,195,175,169,179]
}
df1 = pd.DataFrame(dict1)
print(df1)
liter = df1[['L']]
kilometer = df1[['KM']]
X_train, X_test, y_train, y_test = ms.train_test_split(liter, kilometer, test_size=0.2, random_state=0)
plt.scatter(X_train, y_train, edgecolors='r')
model1 = lm.LinearRegression()
model1.fit(X_train, y_train)
plt.xlabel('liter')
plt.ylabel('Kilometer')
plt.title('Konsumsi Bahan Bakar')
x1 = np.linspace(0,45)
y1 = 24.33 + 5.98 * x1
jarak = model1.predict([[60]]) #bensin
print(jarak)
plt.plot(x1,y1)
plt.show()
mod = lm.LinearRegression()
mod.fit(X_train, y_train)
