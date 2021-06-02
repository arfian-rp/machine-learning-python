import pandas as pd
import sklearn.model_selection as ms
import sklearn.naive_bayes as nb
import sklearn.metrics as met
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df.describe()

df.isnull().sum()

df.drop(['Cabin'], axis=1, inplace=True)

df['Embarked'].value_counts()

df['Embarked'].fillna('S', inplace=True)

embarked = {"Embarked": {"S":0, "C":1, "Q":2}}
df.replace(embarked, inplace=True)

df.dropna(inplace=True, how='any')

df['Fare'] = df['Fare'].astype(int)
df['Age'] = df['Age'].astype(int)

df = df.drop(['PassengerId','Name','Ticket'], axis=1)

sex = {"Sex": {"male":0, "female":1}}
df.replace(sex, inplace=True)

fitur = df[['Pclass','Embarked','Sex','Age','Fare','SibSp','Parch']]
label = df['Survived']
X_train, X_test, y_train, y_test = ms.train_test_split(fitur, label, test_size=0.25, random_state=0)

gnb = nb.GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
akurasi = met.accuracy_score(y_test, y_pred)
presisi = met.precision_score(y_test, y_pred)
print('akurasi : {} , presisi : {}'.format(akurasi, presisi))

y_pred_proba = gnb.predict_proba(X_test)[::,1]
fp,tp,_ = met.roc_curve(y_test, y_pred_proba)
auc = met.roc_auc_score(y_test, y_pred_proba)
plt.plot(fp,tp,label='data 1, auc='+str(auc))
plt.legend(loc=4)
plt.show()