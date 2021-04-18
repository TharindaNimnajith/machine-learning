import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

model = LinearRegression()
boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

X = boston[['RM']]
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

model.fit(X_train, Y_train)

y_test_predicted = model.predict(X_test)

residuals = Y_test - y_test_predicted

plt.scatter(X_test, residuals)
plt.hlines(y=0, xmin=X_test.min(), xmax=X_test.max(), linestyle='--')
plt.xlim((4, 9))
plt.xlabel('RM')
plt.ylabel('residuals')
plt.savefig('plots/residuals.png')
# plt.show()

print(residuals[:5])
print(residuals.mean())
print((residuals ** 2).mean())
print(mean_squared_error(Y_test, y_test_predicted))

print(model.score(X_test, Y_test))
print(((Y_test - Y_test.mean()) ** 2).sum())
print((residuals ** 2).sum())
