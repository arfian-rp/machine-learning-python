import pandas as pd
import sklearn.model_selection as ms
import sklearn.naive_bayes as nb
import sklearn.metrics as met
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df = df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
sex = {'Sex': {'male':0, 'female':1}}
df.replace(sex, inplace=True)
df['Embarked'].fillna('S', inplace=True)
embarked = {'Embarked': {'S':0, 'C':1, 'Q':2}}
df.replace(embarked, inplace=True)
df.head(10)

pria = df['Age'].loc[df['Sex'] == 0].mean()
wanita = df['Age'].loc[df['Sex'] == 1].mean()
print('pria : {}, wanita : {}'.format(pria, wanita))

df['Age'].loc[(df['Sex'] == 0) & (df['Age'].isnull() == True)] = 30.726

df['Age'].loc[(df['Sex'] == 1) & (df['Age'].isnull() == True)] = 27.915

# df['Age'].hist(bins=20)

# df['Age'].loc[df['Survived'] == 1].hist(bins=40)

umur = [0,5,15,25,30,35,45,50,200]
umur_label = ['0-5','5-15','15-25','25-30','30-35','35-45','45-50','>50']
kel_umur = pd.cut(df['Age'], umur, labels=umur_label)
df['KelompokUmur'] = kel_umur
df['KelompokUmurKode'] = df['KelompokUmur'].cat.codes
df['KelompokUmur'].value_counts()

# df['Fare'].hist(bins=20)

harga = [0,10,30,35,80,1000]
harga_label = ['0-10','10-30','30-35','35-80','>80']
kel_harga = pd.cut(df['Fare'], harga, labels=harga_label)
df['KelompokHarga'] = kel_harga
df['KelompokHargaKode'] = df['KelompokHarga'].cat.codes

Jo = df['SibSp'].astype(int) + df['Parch'].astype(int) + 1
df['JumlahOrang'] = Jo.astype(int)

fitur = df[['Pclass','Embarked','Sex','KelompokUmurKode','KelompokHargaKode','JumlahOrang']]
label = df['Survived']
X_train, X_test, y_train, y_test = ms.train_test_split(fitur, label, test_size=0.25, random_state=0)

gnb = nb.GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
akurasi = met.accuracy_score(y_test, y_pred)
presisi = met.precision_score(y_test, y_pred)
print('akurasi = {}, presisi = {}'.format(akurasi, presisi))

y_pred_proba = gnb.predict_proba(X_test)[::,1]
fp,tp,_ = met.roc_curve(y_test, y_pred_proba)
auc = met.roc_auc_score(y_test, y_pred_proba)
plt.plot(fp,tp,label='data 1, auc='+str(auc))
plt.legend(loc=4)
plt.show()