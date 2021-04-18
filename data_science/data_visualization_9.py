import matplotlib.pyplot as plt
import pandas as pd

presidents_df = pd.read_csv('csv_files/president_heights_party.csv', index_col='name')
plt.style.use('classic')
presidents_df.boxplot(column='height')
plt.savefig('plots/boxplot.png')
# plt.draw()
# plt.pause(1)
# plt.show()
# plt.clf()
