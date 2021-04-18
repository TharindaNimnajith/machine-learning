import matplotlib.pyplot as plt
import pandas as pd

presidents_df = pd.read_csv('csv_files/president_heights_party.csv', index_col='name')
plt.hist(presidents_df['height'], bins=5)
plt.savefig('plots/hist_2.png')
# plt.draw()
# plt.pause(1)
# plt.show()
# plt.clf()
