import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston

boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data,
                      columns=boston_dataset.feature_names)

boston['MEDV'] = boston_dataset.target

boston.hist(column='CHAS', bins=20)
plt.savefig('plots/boston_2.png')
# plt.show()
