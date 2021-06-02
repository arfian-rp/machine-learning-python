import pandas as pd
import sklearn.model_selection as ms
import sklearn.neural_network as nn
import sklearn.preprocessing as pp
import sklearn.metrics as met

df = pd.read_csv('datatraining.csv', header=0, names=['id','date','Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy'])

X = df.drop(['id','date','Occupancy'], axis=1)
y = df['Occupancy']

X.describe()

X_train, X_test, y_train, y_test = ms.train_test_split(X,y,test_size=0.2)

X_train.count()

scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
scl.fit(X_train)
X_train = scl.transform(X_train)
X_test = scl.transform(X_test)

print('min = {}, max = {}'.format(X_train.min(), X_train.max()))

mlp = nn.MLPClassifier(hidden_layer_sizes=(1), max_iter=5)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print(met.classification_report(y_test, y_pred))

print(X_test)