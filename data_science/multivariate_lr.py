import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

X = boston[['RM']]
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

model = LinearRegression()
model.fit(X_train, Y_train)

y_test_predicted = model.predict(X_test)
print(y_test_predicted)
print(Y_test)

X2 = boston[['RM', 'LSTAT']]
Y2 = boston['MEDV']

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.3, random_state=1)

model2 = LinearRegression()
model2.fit(X2_train, Y2_train)

y2_test_predicted = model2.predict(X2_test)
print(y2_test_predicted)
print(Y2_test)

print(model.intercept_)
print(model.coef_)

print(model2.intercept_)
print(model2.coef_)

print(mean_squared_error(Y_test, y_test_predicted))
print(mean_squared_error(Y2_test, y2_test_predicted))
