import graphviz
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz

df = pd.read_csv('csv_files/titanic.csv')

df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print(model.predict([[3, True, 22, 1, 0, 7.25]]))
print()

kf = KFold(n_splits=5, shuffle=True, random_state=10)

dt_accuracy_scores = []
dt_precision_scores = []
dt_recall_scores = []

lr_accuracy_scores = []
lr_precision_scores = []
lr_recall_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_accuracy_scores.append(dt.score(X_test, y_test))
    dt_y_pred = dt.predict(X_test)
    dt_precision_scores.append(precision_score(y_test, dt_y_pred))
    dt_recall_scores.append(recall_score(y_test, dt_y_pred))
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_accuracy_scores.append(lr.score(X_test, y_test))
    lr_y_pred = lr.predict(X_test)
    lr_precision_scores.append(precision_score(y_test, lr_y_pred))
    lr_recall_scores.append(recall_score(y_test, lr_y_pred))

print('Decision Tree:')
print('Accuracy:', np.mean(dt_accuracy_scores))
print('Precision:', np.mean(dt_precision_scores))
print('Recall:', np.mean(dt_recall_scores))
print()

print('Logistic Regression:')
print('Accuracy:', np.mean(lr_accuracy_scores))
print('Precision:', np.mean(lr_precision_scores))
print('Recall:', np.mean(lr_recall_scores))
print()

kf = KFold(n_splits=5, shuffle=True)

for criterion in ['gini', 'entropy']:
    print('Decision Tree - {}'.format(criterion))
    accuracy = []
    precision = []
    recall = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dt = DecisionTreeClassifier(criterion=criterion)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
    print('Accuracy:', np.mean(accuracy))
    print('Precision:', np.mean(precision))
    print('Recall:', np.mean(recall))
    print()

feature_names = ['Pclass', 'male']

X = df[feature_names].values
y = df['Survived'].values

dt = DecisionTreeClassifier()
dt.fit(X, y)

dot_file = export_graphviz(dt, feature_names=feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename='plots/decision_tree_1', format='png', cleanup=True)

dt1 = DecisionTreeClassifier(max_depth=2, min_samples_leaf=2, max_leaf_nodes=10)
dt1.fit(X, y)

dot_file = export_graphviz(dt1, feature_names=feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename='plots/decision_tree_2', format='png', cleanup=True)

param_grid = {
    'max_depth': [5, 15, 25],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 35, 50]
}

dt2 = DecisionTreeClassifier()

gs = GridSearchCV(dt2, param_grid, scoring='f1', cv=5)
gs.fit(X, y)

print('Best params:', gs.best_params_)
print('Best score:', gs.best_score_)
