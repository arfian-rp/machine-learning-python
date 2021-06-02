import vega_datasets as vd
import numpy as np
import sklearn.linear_model as lm
import seaborn as sn
import matplotlib.pyplot as plt

iris = vd.data.iris()
sn.pairplot(iris, hue='species')
plt.show()
iris = iris.drop(iris[iris['species'] == 'versicolor'].index)
categ = {'setosa':0, 'versicolor':2, 'virginica':1}
color = {'setosa':'blue', 'versicolor':'red', 'virginica':'green'}
plt.scatter(iris['petalWidth'], [categ[r] for r in iris['species']], c=[color[r] for r in iris['species']])
plt.show()
clf = lm.LogisticRegression()
clf.fit(iris[['petalWidth']], iris['species'])

print(clf.predict([[1.2]]))