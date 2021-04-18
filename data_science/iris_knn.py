import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

cross_validation_scores = cross_val_score(knn, X, y, cv=5)
print(cross_validation_scores)
print(cross_validation_scores.mean())

knn2 = KNeighborsClassifier()

param_grid = {
    'n_neighbors': np.arange(2, 10)
}

knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
knn_gscv.fit(X, y)

print(knn_gscv.best_params_)
print(knn_gscv.best_score_)

knn_final = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
knn_final.fit(X, y)

y_pred = knn_final.predict(X)
print(knn_final.score(X, y))

new_data = np.array([3.76, 1.20])
print(new_data)

new_data = new_data.reshape(1, -1)
print(new_data)

new_data = np.array([[3.76, 1.20]])
print(new_data)

print(knn_final.predict(np.array(new_data)))
