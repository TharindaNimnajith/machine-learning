import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

boston_dataset = load_boston()
print(type(boston_dataset))

boston = pd.DataFrame(boston_dataset.data,
                      columns=boston_dataset.feature_names)

boston['MEDV'] = boston_dataset.target

print(boston.shape)
print(boston.columns)

print(boston.head())
print(boston.tail())

print(boston.describe().round(2))
print(boston.describe(percentiles=[0.25, 0.75]).round(2))

boston.hist(column='CHAS')
plt.savefig('plots/boston_1.png')
# plt.show()

corr_matrix = boston.corr().round(2)
print(corr_matrix)

boston.plot(kind='scatter',
            x='RM',
            y='MEDV',
            figsize=(8, 6))
plt.savefig('plots/scatter_boston_1.png')
# plt.show()

boston.plot(kind='scatter',
            x='LSTAT',
            y='MEDV',
            figsize=(8, 6))
plt.savefig('plots/scatter_boston_2.png')
# plt.show()

X = boston[['RM']]
print(X.shape)

Y = boston['MEDV']
print(Y.shape)

model = LinearRegression()
print(model)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

model.fit(X_train, Y_train)

print(model.intercept_.round(2))
print(model.coef_.round(2))

new_RM = np.array([6.5]).reshape(-1, 1)
print(model.predict(new_RM))
print(model.intercept_ + model.coef_ * 6.5)

y_test_predicted = model.predict(X_test)
print(y_test_predicted.shape)
print(type(y_test_predicted))
