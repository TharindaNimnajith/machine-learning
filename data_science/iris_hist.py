import matplotlib.pyplot as plt
import pandas as pd

iris = pd.read_csv('csv_files/iris.csv')

iris['sepal_wd'].hist()
print(iris.hist())
plt.savefig('plots/iris_sep_wid_hist.png')
plt.show()
