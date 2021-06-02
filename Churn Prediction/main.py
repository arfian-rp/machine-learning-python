import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.linear_model as lm
import sklearn.ensemble as en
import sklearn.metrics as met
import joblib as jb

df = pd.read_csv('churnprediction_ch9.csv', sep=',', index_col=['customer_id'])

aktif = df.groupby('churn').count()

plt.pie(aktif['product'],labels=['Aktif','Churn'], autopct='%1.0f%%')
plt.axis('equal')
plt.show()

df['product'].value_counts()

data = pd.concat([df, pd.get_dummies(df['product'])], axis=1, sort=False)
data.drop(['product'], axis=1, inplace=True)

dfk = data.corr()
sns.heatmap(dfk, xticklabels=dfk.columns.values, yticklabels=dfk.columns.values, annot=True, annot_kws={'size':12})
heat_map = plt.gcf()
heat_map.set_size_inches(10,10)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

data.head(3)

X = data.drop(['reload_2','socmed_2','games','churn'], axis=1, inplace=False)
y = data['churn']

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=0)

scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
scl.fit(X_train)
X_train = scl.transform(X_train)
X_test = scl.transform(X_test)

# ALGORITMA LOGISTIC REGRESSION


model = lm.LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

print(' ')
print('ALGORITMA LOGISTIC REGRESSION')
y_pred = model.predict(X_test)
print('y_pred : ', y_pred)
score = met.accuracy_score(y_test, y_pred)
print('score : ', score)
presisi = met.precision_score(y_test, y_pred)
print('presisi : ', presisi)
recall = met.recall_score(y_test, y_pred)
print('recall : ', recall)
auc = met.roc_auc_score(y_test, y_pred)
print('auc : ', auc)
print(' ')

jb.dump(model, "modelLogistic.sav")

# ALGORITMA RANDOM FOREST

model = en.RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X_train, y_train)

print('ALGORITMA RANDOM FOREST')
y_pred = model.predict(X_test)
print('y_pred : ', y_pred)
score = met.accuracy_score(y_test, y_pred)
print('score : ', score)
presisi = met.precision_score(y_test, y_pred)
print('presisi : ', presisi)
recall = met.recall_score(y_test, y_pred)
print('recall : ', recall)
auc = met.roc_auc_score(y_test, y_pred)
print('auc : ', auc)

jb.dump(model, "modelRF.sav")

fitur_penting = pd.Series(model.feature_importances_, index=X.columns)
fitur_penting.nlargest(10).plot(kind='barh')
plt.show()
