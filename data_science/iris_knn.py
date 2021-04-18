import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv('csv_files/iris.csv')

iris.drop('id', axis=1, inplace=True)

X = iris[['petal_len', 'petal_wd']]
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(knn)

pred = knn.predict(X_test)
print(pred[:5])

y_pred_prob = knn.predict_proba(X_test)
print(y_pred_prob[10:12])
print(pred[10:12])

correct = (pred == y_test.values).sum()
print(correct)

total = y_test.size
print(total)

accuracy = correct / total
print(accuracy)

print(knn.score(X_test, y_test))
print(accuracy_score(y_test, pred))

print(confusion_matrix(y_test, pred))
print(confusion_matrix(y_test, pred, labels=['iris-setosa', 'iris-versicolor', 'iris-verginica']))
print(plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues))
plt.savefig('plots/confusion_matrix.png')
plt.show()
