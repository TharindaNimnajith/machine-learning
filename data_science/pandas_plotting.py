import pandas as pd

iris = pd.read_csv('csv_files/iris.csv')
iris.drop('id', axis=1, inplace=True)

print(pd.plotting.scatter_matrix(iris))
