import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as met
import sklearn.feature_selection as fs
import matplotlib.pyplot as plt

df = pd.read_csv('calonpembeli.csv')
df.describe()

df = df[df['Usia'] <= 100]

df.isnull().sum()
df.dropna()

df['Beli_Mobil'].value_counts()

X = df[['Usia', 'Status', 'Kelamin', 'Memiliki_Mobil', 'Penghasilan']]
y = df.Beli_Mobil
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=0)

model = lm.LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

print(model.coef_)

y_pred = model.predict(X_test)
print(y_pred)

X_test.head()

y_test.head(1)

confusionmatix = met.confusion_matrix(y_test, y_pred)
print(confusionmatix)

score = model.score(X_test, y_test)
print(score)

presisi = met.precision_score(y_test, y_pred)
print(presisi)

auc = met.roc_auc_score(y_test, y_pred)
print(auc)

y_pred_proba = model.predict_proba(X_test)[::,1]
fp, tp, _ = met.roc_curve(y_test, y_pred_proba)
auc = met.roc_auc_score(y_test, y_pred_proba)
plt.plot(fp,tp,label='data 1, auc = '+str(auc))
plt.legend(loc=4)
plt.show()

#FEATURE SELECTION DENGAN METHOD RFE (Recursive Feature Elimination)
rfe = fs.RFE(model, 3)
rfe.fit(X_train, y_train)
print('Suport = ', rfe.support_)
print('Ranking = ', rfe.ranking_)