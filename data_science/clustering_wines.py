import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

x1 = np.array([0, 1])
x2 = np.array([2, 0])

print(np.sqrt(((x1 - x2) ** 2).sum()))
print(np.sqrt(5))

data = load_wine()
wine = pd.DataFrame(data.data, columns=data.feature_names)

print(wine.shape)
print(wine.columns)
print(wine.info())
print(wine.describe())
print(wine.iloc[:, :5].describe())
print(wine.head())
print(wine.tail())

scatter_matrix(wine.iloc[:, [0, 5]])
plt.savefig('plots/wine_scatter.png')
plt.show()

X = wine[['alcohol', 'total_phenols']]

scale = StandardScaler()
scale.fit(X)
print(scale.mean_)
print(scale.scale_)

X_scaled = scale.transform(X)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))
