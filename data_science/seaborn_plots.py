import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

iris = pd.read_csv('csv_files/iris.csv')
iris.drop('id', axis=1, inplace=True)

sns.catplot(x='sepal_wd', y='sepal_len', hue='species', data=iris)
plt.savefig('plots/sns_plots_1.png')
plt.show()

sns.pairplot(iris)
plt.savefig('plots/sns_plots_2.png')
plt.show()
