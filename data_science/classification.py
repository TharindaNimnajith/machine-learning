# from sklearn.datasets import load_iris
# import pandas as pd
#
# iris = load_iris()
# print(type(iris))
# iris = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(type(iris))
# print(iris.shape)

import pandas as pd
import matplotlib.pyplot as plt

# iris = pd.read_csv('https://sololearn.com/uploads/files/iris.csv')
iris = pd.read_csv('csv_files/iris.csv')
print(iris.shape)
print(iris.head())

iris.drop('id', axis=1, inplace=True)
print(iris.head())
print(iris.describe())
print(iris[['petal_len','petal_wd']].describe())
print(iris.groupby('species').size())
print(iris['species'].value_counts())

iris.hist()
print(iris.hist())
plt.savefig('plots/iris_hist.png')
plt.show()
