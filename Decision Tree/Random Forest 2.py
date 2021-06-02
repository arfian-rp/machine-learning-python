import pandas as pd
import numpy as np
import sklearn.tree as tree
import sklearn.model_selection as ms
import sklearn.ensemble as en
import sklearn.metrics as met
import sklearn.preprocessing as prep

df = pd.read_csv('bab6_adult_data.csv')
df.head()

df.dropna(inplace=True)

gender = prep.LabelEncoder()
gender.fit(df['sex'])
df['sex_code'] = gender.transform(df['sex'])

edu = prep.LabelEncoder()
edu.fit(df['education'])
df['education_code'] = edu.transform(df['education'])

race = prep.LabelEncoder()
race.fit(df['race'])
df['race_code'] = race.transform(df['race'])

workclass = prep.LabelEncoder()
workclass.fit(df['workclass'])
df['workclass_code'] = workclass.transform(df['workclass'])

occupation = prep.LabelEncoder()
occupation.fit(df['occupation'])
df['occupation_code'] = occupation.transform(df['occupation'])

relationship = prep.LabelEncoder()
relationship.fit(df['relationship'])
df['relationship_code'] = relationship.transform(df['relationship'])

native = prep.LabelEncoder()
native.fit(df['native_country'])
df['native_code'] = native.transform(df['native_country'])

X = df.drop(['fn/wft','workclass','education','marital_status','occupation','relationship','race','sex','native_country','label'], axis=1)
y = df['label']

X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=0.2, random_state=0)

rf = en.RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print('pred : ',y_pred)

akurasi = met.accuracy_score(y_test, y_pred)
print('akurasi : ',akurasi)

print(met.classification_report(y_test, y_pred))
print(rf.feature_importances_)