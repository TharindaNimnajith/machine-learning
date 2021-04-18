import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston

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
