import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

cancer_data = load_breast_cancer()

df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']
print(df)

X = df[cancer_data.feature_names].values
y = df['target'].values
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

first_row = X_test[0]

print(rf.predict([first_row])[0])
print(y_test[0])

print(rf.score(X_test, y_test))

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

print(dt.score(X_test, y_test))

param_grid = {
    'n_estimators': [10, 25, 50, 75, 100]
}

rf1 = RandomForestClassifier(max_features=5, n_estimators=15, random_state=123)

gs = GridSearchCV(rf1, param_grid, scoring='f1', cv=5)
gs.fit(X, y)

print(gs.best_params_)
print(gs.best_score_)
print(gs.best_index_)
print(gs.best_estimator_)

n_estimators = list(range(1, 101))

param_grid = {
    'n_estimators': n_estimators
}

rf2 = RandomForestClassifier()

gs2 = GridSearchCV(rf, param_grid, cv=5)
gs2.fit(X, y)
scores = gs2.cv_results_['mean_test_score']

plt.plot(n_estimators, scores)
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xlim(0, 100)
plt.ylim(0.9, 1)
plt.savefig('plots/random_forest_elbow_graph.png')
plt.show()

rf3 = RandomForestClassifier(n_estimators=10)
rf3.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

rf4 = RandomForestClassifier(n_estimators=10, random_state=111)
rf4.fit(X_train, y_train)

ft_imp = pd.Series(rf4.feature_importances_, index=cancer_data.feature_names).sort_values(ascending=False)
print(ft_imp.head(10))
