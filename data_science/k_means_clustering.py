import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

data = load_wine()
wine = pd.DataFrame(data.data, columns=data.feature_names)

X = wine[['alcohol', 'total_phenols']]

scale = StandardScaler()
scale.fit(X)
X_scaled = scale.transform(X)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)

y_pred = kmeans.predict(X_scaled)
print(y_pred)

print(kmeans.cluster_centers_)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=250, c=[0, 1, 2],
            edgecolors='k')
plt.xlabel('alcohol')
plt.ylabel('total phenols')
plt.title('k-means (k=3)')
plt.savefig('plots/k_means.png')
plt.show()

X_new = np.array([[13, 2.5]])
X_new_scaled = scale.transform(X_new)
print(kmeans.predict(X_new_scaled))

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_scaled)
print(kmeans.inertia_)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
print(kmeans.inertia_)

inertia = []

for i in np.arange(1, 11):
    km = KMeans(n_clusters=i)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(np.arange(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('plots/inertia.png')
plt.show()
