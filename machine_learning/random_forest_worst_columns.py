import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

cancer_data = load_breast_cancer()

df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']

X = df[cancer_data.feature_names].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))

worst_cols = [col for col in df.columns if 'worst' in col]
print(worst_cols)
X_worst = df[worst_cols]

rf1 = RandomForestClassifier(n_estimators=10, random_state=111)
rf1.fit(X_train, y_train)
print(rf1.score(X_test, y_test))
