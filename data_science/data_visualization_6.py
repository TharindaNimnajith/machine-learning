import matplotlib.pyplot as plt
import pandas as pd

presidents_df = pd.read_csv('csv_files/president_heights_party.csv', index_col='name')
presidents_df.plot(kind='scatter',
                   x='height',
                   y='age',
                   title='U.S. presidents')
plt.savefig('plots/pd_plot.png')
# plt.draw()
# plt.pause(1)
# plt.show()
# plt.clf()
