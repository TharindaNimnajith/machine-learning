import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

df = pd.read_csv('csv_files/titanic.csv')

X = df[['Age', 'Fare']].values[:6]
y = df['Survived'].values[:6]

scores = []

kf = KFold(n_splits=5, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

print(scores)
print(np.mean(scores))

final_model = LogisticRegression()
final_model.fit(X, y)
