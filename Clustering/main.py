import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cls

df = pd.read_csv('penerbangan1.csv')
X = df['Waktu']
y = df['Jarak']
plt.scatter(X,y)
plt.show()

dfk = df[['Waktu','Jarak']]
kMeans = cls.KMeans(n_clusters=2)
kMeans.fit(dfk)

centroids = kMeans.cluster_centers_
print(centroids)

cen_X = centroids[:,0]
cen_y = centroids[:,1]

plt.scatter(cen_X, cen_y, color='black', marker='X', s=100)

label = kMeans.labels_
print(label)

plt.scatter(X, y, cmap='rainbow')
plt.title("clustering pakai KMeans")
plt.show()

print(kMeans.inertia_)

jarak_total = []
K = range(1,10)
for k in K:
  kMeans = cls.KMeans(n_clusters=k)
  kMeans.fit(dfk)
  jarak_total.append(kMeans.inertia_)
plt.plot(K, jarak_total)
plt.xlabel('k')
plt.show()
