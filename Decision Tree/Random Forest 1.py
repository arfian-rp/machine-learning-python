import pandas as pd
import sklearn.ensemble as en
import sklearn.model_selection as ms
import sklearn.metrics as met

df = pd.read_csv('decisiontree_ch6.csv')
encoding = {'mesin': {'bensin':0, 'diesel':1},
            'penggerak': {'depan':0, 'belakang':1}}
df.replace(encoding, inplace=True)
print(df)

X = df.drop(['ID', 'label'], axis=1)
y = df['label']
X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=0.2)

rf = en.RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print('pred : ', y_pred)

akurasi = met.accuracy_score(y_test, y_pred)
print('akurasi : ',akurasi)

score = rf.feature_importances_
print('score : ', score)