import matplotlib.pyplot as plt
import pandas as pd

# Matplotlib is a library used to create graphs, charts, and figures.
# It also provides functions to customize your figures by changing the colors and labels.
help(plt)
print(dir(plt))
print(plt)

s = pd.Series([18, 42, 9, 32, 81, 64, 3])
s.plot(kind='bar')
plt.savefig('plot.png')

df = pd.read_csv('./ca-covid.csv')
df.drop('state', axis=1, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%y')
df['month'] = df['date'].dt.month
df.set_index('date', inplace=True)

df[df['month'] == 12]['cases'].plot()
plt.savefig('cases.png')

df[df['month'] == 12][['cases', 'deaths']].plot()
plt.savefig('cases_deaths.png')

df.groupby('month')['cases'].sum().plot(kind='bar')
plt.savefig('cases_per_month.png')

df.groupby('month')['deaths'].sum().plot(kind='bar')
plt.savefig('deaths_per_month.png')

df.groupby('month')[['cases', 'deaths']].sum().plot(kind='bar', stacked=True)
plt.savefig('cases_deaths_per_month.png')

df.groupby('month')[['cases', 'deaths']].sum().plot(kind='barh', stacked=True)
plt.savefig('cases_deaths_per_month_horizontal.png')

df[df['month'] == 6]['cases'].plot(kind='box')
plt.savefig('boxplot.png')

df[df['month'] == 6]['cases'].plot(kind='hist')
plt.savefig('histogram_1.png')

df[df['month'] == 6]['cases'].plot(kind='hist', bins=10)
plt.savefig('histogram_2.png')

df[df['month'] == 6][['cases', 'deaths']].plot(kind='area', stacked=False)
plt.savefig('area_not_stacked.png')

df[df['month'] == 6][['cases', 'deaths']].plot(kind='area')
plt.savefig('area_stacked.png')

df[df['month'] == 6][['cases', 'deaths']].plot(kind='scatter', x='cases', y='deaths')
plt.savefig('scatter.png')

df.groupby('month')['cases'].sum().plot(kind='pie')
plt.savefig('pie.png')
