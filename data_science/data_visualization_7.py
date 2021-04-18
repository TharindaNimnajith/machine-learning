import matplotlib.pyplot as plt
import pandas as pd

presidents_df = pd.read_csv('csv_files/president_heights_party.csv', index_col='name')
presidents_df['height'].plot(kind='hist',
                             title='height',
                             bins=5)
plt.savefig('plots/hist_1.png')
# plt.draw()
# plt.pause(1)
# plt.show()
# plt.clf()
