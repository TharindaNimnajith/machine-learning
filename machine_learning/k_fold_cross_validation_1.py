import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

df = pd.read_csv('csv_files/titanic.csv')

X = df[['Age', 'Fare']].values[:6]
y = df['Survived'].values[:6]

kf = KFold(n_splits=3, shuffle=True)

splits = list(kf.split(X))
print(splits)

first_split = splits[0]
train_indices, test_indices = first_split
print('Training set indices:', train_indices)
print('Test set indices:', test_indices)

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]
print(X_train)
print(y_train)
print(X_test)
print(y_test)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
