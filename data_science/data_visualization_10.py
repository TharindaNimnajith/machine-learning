import matplotlib.pyplot as plt
import pandas as pd

presidents_df = pd.read_csv('csv_files/president_heights_party.csv', index_col='name')
party_cnt = presidents_df['party'].value_counts()
print(party_cnt)
plt.style.use('ggplot')
party_cnt.plot(kind='bar')
plt.savefig('plots/bar.png')
# plt.draw()
# plt.pause(1)
# plt.show()
# plt.clf()
