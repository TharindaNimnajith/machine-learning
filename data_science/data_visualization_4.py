import matplotlib.pyplot as plt
import pandas as pd

presidents_df = pd.read_csv('csv_files/president_heights_party.csv', index_col='name')
plt.scatter(presidents_df['height'], presidents_df['age'])
plt.savefig('plots/presidents_scatter.png')
# plt.draw()
# plt.pause(1)
# plt.show()
# plt.clf()
