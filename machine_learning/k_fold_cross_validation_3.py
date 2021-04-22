import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

df = pd.read_csv('csv_files/titanic.csv')

df['male'] = df['Sex'] == 'male'

kf = KFold(n_splits=5, shuffle=True)

X1 = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X2 = df[['Pclass', 'male', 'Age']].values
X3 = df[['Fare', 'Age']].values

y = df['Survived'].values


def score_model(X, y, kf):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    print('Accuracy:', np.mean(accuracy_scores))
    print('Precision:', np.mean(precision_scores))
    print('Recall:', np.mean(recall_scores))
    print('F1 Score:', np.mean(f1_scores))


print('Logistic Regression with all features:')
score_model(X1, y, kf)
print()

print('Logistic Regression with Pclass, Sex & Age features:')
score_model(X2, y, kf)
print()

print('Logistic Regression with Fare & Age features:')
score_model(X3, y, kf)
print()

model = LogisticRegression()
model.fit(X1, y)

print(model.predict([[3, False, 25, 0, 1, 2]]))
