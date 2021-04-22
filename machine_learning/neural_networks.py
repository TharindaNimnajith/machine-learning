from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=3)
print(X)
print(y)

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=100, edgecolors='k')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=100, edgecolors='k', marker='^')
plt.savefig('plots/make_classification.png')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

mlp = MLPClassifier()
mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))

mlp1 = MLPClassifier(max_iter=1000)
mlp1.fit(X_train, y_train)
print(mlp1.score(X_test, y_test))

mlp2 = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50))
mlp2.fit(X_train, y_train)
print(mlp2.score(X_test, y_test))

mlp3 = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50), alpha=0.0001, solver='adam', random_state=3)
mlp3.fit(X_train, y_train)
print(mlp3.score(X_test, y_test))
