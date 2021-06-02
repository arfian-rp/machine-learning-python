import vega_datasets as vd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.naive_bayes as nb
import sklearn.model_selection as ms
import sklearn.metrics as met

iris = vd.data.iris()
spc = {'setosa':1, 'versicolor':2, 'virginica':3}
iris['spc'] = [spc[r] for r in iris['species']]

X = iris[['sepalWidth','sepalLength','petalWidth','petalLength']]
y = iris['spc']

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)

clf = nb.GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(y_pred == y_test)

print(met.r2_score(y_test, y_pred))