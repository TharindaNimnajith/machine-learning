from sklearn.datasets import make_classification
from matplotlib import pyplot as plt

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=3)
print(X)
print(y)

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=100, edgecolors='k')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=100, edgecolors='k', marker='^')
plt.savefig('plots/make_classification.png')
plt.show()
