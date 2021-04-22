import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
