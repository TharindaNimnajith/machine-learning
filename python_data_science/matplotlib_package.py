import matplotlib.pyplot as plt
import pandas as pd

# Matplotlib is a library used to create graphs, charts, and figures.
# It also provides functions to customize your figures by changing the colors and labels.
help(plt)
print(dir(plt))
print(plt)

s = pd.Series([18, 42, 9, 32, 81, 64, 3])
s.plot(kind='bar')
plt.savefig('./plots/plot.png')

df = pd.read_csv('csv_files/ca-covid.csv')
df.drop('state', axis=1, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%y')
df['month'] = df['date'].dt.month
df.set_index('date', inplace=True)

df[df['month'] == 12]['cases'].plot()
plt.savefig('./plots/cases.png')

df[df['month'] == 12][['cases', 'deaths']].plot()
plt.savefig('./plots/cases_deaths.png')

df.groupby('month')['cases'].sum().plot(kind='bar')
plt.savefig('./plots/cases_per_month.png')

df.groupby('month')['deaths'].sum().plot(kind='bar')
plt.savefig('./plots/deaths_per_month.png')

df.groupby('month')[['cases', 'deaths']].sum().plot(kind='bar', stacked=True)
plt.savefig('./plots/cases_deaths_per_month.png')

df.groupby('month')[['cases', 'deaths']].sum().plot(kind='barh', stacked=True)
plt.savefig('./plots/cases_deaths_per_month_horizontal.png')

df[df['month'] == 6]['cases'].plot(kind='box')
plt.savefig('./plots/boxplot.png')

df[df['month'] == 6]['cases'].plot(kind='hist')
plt.savefig('./plots/histogram_1.png')

df[df['month'] == 6]['cases'].plot(kind='hist', bins=10)
plt.savefig('./plots/histogram_2.png')

df[df['month'] == 6][['cases', 'deaths']].plot(kind='area', stacked=False)
plt.savefig('./plots/area_not_stacked.png')

df[df['month'] == 6][['cases', 'deaths']].plot(kind='area')
plt.savefig('./plots/area_stacked.png')

df[df['month'] == 6][['cases', 'deaths']].plot(kind='scatter', x='cases', y='deaths')
plt.savefig('./plots/scatter.png')

df.groupby('month')['cases'].sum().plot(kind='pie')
plt.savefig('./plots/pie.png')

df = df[df['month'] == 6]
df[['cases', 'deaths']].plot(kind='line', legend=True, stacked=False, color=['#1970E7', '#E73E19'])
plt.xlabel('Days in June')
plt.ylabel('Number')
plt.suptitle('COVID-19 in June')
plt.savefig('./plots/legend.png')
