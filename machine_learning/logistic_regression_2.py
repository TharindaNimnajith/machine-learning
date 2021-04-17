import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('csv_files/titanic.csv')

df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(X, y)

# save the model to disk
filename = './models/logistic_regression_2.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model)

print(model.predict([[3, True, 22.0, 1, 0, 7.25]]))

print(model.predict(X[:5]))
print(y[:5])

print(model.coef_, model.intercept_)
# [[-1.1364554  -2.6440985  -0.04237387 -0.38746592 -0.09619824  0.00297188]] [5.08856788]

print(model.coef_[0, 0])  # -1.1364554045737778
print(model.coef_[0, 1])  # -2.6440985
print(model.coef_[0, 2])  # -0.04237387
print(model.coef_[0, 3])  # -0.38746592
print(model.coef_[0, 4])  # 0.09619824
print(model.coef_[0, 5])  # 0.00297188
print(model.intercept_[0])  # 5.08856788

y_pred = model.predict(X)
print((y == y_pred).sum())
print(y.shape[0])
print((y == y_pred).sum() / y.shape[0])
print(model.score(X, y))
