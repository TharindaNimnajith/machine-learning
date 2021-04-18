# from sklearn.datasets import load_iris
# import pandas as pd
#
# iris = load_iris()
# print(type(iris))
# iris = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(type(iris))
# print(iris.shape)

import matplotlib.pyplot as plt
import pandas as pd

# iris = pd.read_csv('https://sololearn.com/uploads/files/iris.csv')
iris = pd.read_csv('csv_files/iris.csv')
print(iris.shape)
print(iris.head())

iris.drop('id', axis=1, inplace=True)
print(iris.head())
print(iris.describe())
print(iris[['petal_len', 'petal_wd']].describe())
print(iris.groupby('species').size())
print(iris['species'].value_counts())

iris.hist()
print(iris.hist())
plt.savefig('plots/iris_hist.png')
plt.show()

inv_name_dict = {
    'iris-setosa': 0,
    'iris-versicolor': 1,
    'iris-virginica': 2
}

colors = [inv_name_dict[item] for item in iris['species']]

scatter = plt.scatter(iris['sepal_len'], iris['sepal_wd'], c=colors)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend(handles=scatter.legend_elements()[0], labels=inv_name_dict.keys())
plt.savefig('plots/iris_scatter_sepal.png')
plt.show()

scatter = plt.scatter(iris['petal_len'], iris['petal_wd'], c=colors)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend(handles=scatter.legend_elements()[0], labels=inv_name_dict.keys())
plt.savefig('plots/iris_scatter_petal.png')
plt.show()
