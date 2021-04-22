import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

print(X.shape, y.shape)
print(np.min(X), np.max(X))
print(y[0:5])

X5 = X[y <= '3']
y5 = y[y <= '3']

mlp = MLPClassifier(hidden_layer_sizes=(6,), max_iter=200, alpha=1e-4, solver='sgd', random_state=2)
mlp.fit(X5, y5)

print(mlp.coefs_)
print(len(mlp.coefs_))
print(mlp.coefs_[0].shape)

fig, axes = plt.subplots(2, 3, figsize=(5, 4))

for i, ax in enumerate(axes.ravel()):
    coef = mlp.coefs_[0][:, i]
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(i + 1)

plt.savefig('/plots/matshow_6.png')
plt.show()
