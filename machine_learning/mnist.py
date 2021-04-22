import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = load_digits(n_class=2, return_X_y=True)

print(X.shape, y.shape)
print(X[0])
print(y[0])
print(X[0].reshape(8, 8))

plt.matshow(X[0].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.savefig('plots/matshow_1.png')
plt.show()

plt.matshow(X[1].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.savefig('plots/matshow_2.png')
plt.show()

X, y = load_digits(n_class=10, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

mlp = MLPClassifier()
mlp.fit(X_train, y_train)

x = X_test[0]
print(mlp.predict([x]))

plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.savefig('plots/matshow_3.png')
plt.show()

x = X_test[1]
print(mlp.predict([x]))

plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.savefig('plots/matshow_4.png')
plt.show()

print(mlp.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

mlp1 = MLPClassifier(random_state=2)
mlp1.fit(X_train, y_train)

print(mlp1.score(X_test, y_test))

y_pred = mlp1.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

j = 0
print(incorrect[j].reshape(8, 8).astype(int))
print(incorrect_true[j])
print(incorrect_pred[j])

plt.matshow(incorrect[j].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.savefig('plots/matshow_5.png')
plt.show()
