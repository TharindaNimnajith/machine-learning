import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv('csv_files/iris.csv')

iris['sepal_wd'].hist()
print(iris.hist())
plt.savefig('plots/iris_sep_wid_hist.png')
plt.show()
