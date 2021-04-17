import numpy as np
import pandas as pd

print(pd.Series([1, 2, 3], index=['a', 'b', 'c']))
print(pd.Series(np.array([1, 2, 3]), index=['a', 'b', 'c']))
print(pd.Series({'a': 1, 'b': 2, 'c': 3}))
series = pd.Series({'a': 1, 'b': 2, 'c': 3})
print(series['a'])

wine_dict = {
    'red_wine': [3, 6, 5],
    'white_wine': [5, 0, 10]
}
sales = pd.DataFrame(wine_dict, index=['adam', 'bob', 'charles'])
print(sales)
print(sales['white_wine'])

# presidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')
presidents_df = pd.read_csv('csv_files/president_heights_party.csv', index_col='name')
print(presidents_df)
print(presidents_df.size)
print(presidents_df.shape)
print(presidents_df.shape[0])
print(presidents_df.head())
print(presidents_df.head(n=3))
print(presidents_df.tail())
print(presidents_df.tail(n=3))
presidents_df.info()
print(presidents_df.loc['Abraham Lincoln'])
print(type(presidents_df.loc['Abraham Lincoln']))
print(presidents_df.loc['Abraham Lincoln', ['age', 'party']])
print(type(presidents_df.loc['Abraham Lincoln', ['age', 'party']]))
print(presidents_df.loc['Abraham Lincoln'].shape)
print(presidents_df.loc['Abraham Lincoln':'Ulysses S. Grant'])
print(presidents_df.iloc[15])
print(presidents_df.iloc[15:18])
print(presidents_df.columns)
print(presidents_df['height'])
print(presidents_df['height'].shape)
print(presidents_df[['height', 'age']].head(n=3))
print(presidents_df[['height', 'age']].shape)
print(presidents_df.loc[:, 'order':'height'].head(n=3))
print(presidents_df.min())
print(presidents_df.max())
print(presidents_df.mean())
print(presidents_df['age'].min())
print(presidents_df['age'].max())
print(presidents_df['age'].mean())
print(presidents_df['age'].median())
print(presidents_df['age'].quantile(0.5))
print(presidents_df['age'].quantile([0.25, 0.5, 0.75, 1]))
print(presidents_df['age'].std())
print(presidents_df['age'].var())
print(presidents_df.describe())
print(presidents_df['age'].describe())
print(presidents_df['party'].describe())
print(presidents_df['party'].value_counts())
print(presidents_df.groupby('party').mean())
print(presidents_df.groupby('party')['height'].mean())
print(presidents_df.groupby('party')['height'].agg(['min', np.median, max, np.mean, 'count', 'std', 'var']))
print(presidents_df.groupby('party').agg({
    'height': [np.median, np.mean],
    'age': [min, max]
}))
