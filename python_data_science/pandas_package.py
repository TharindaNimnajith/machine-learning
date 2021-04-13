import pandas as pd

help(pd)
print(dir(pd))
print(pd)

# The two primary components of pandas are the Series and the DataFrame.
# A Series is essentially a column, and a DataFrame is a multi-dimensional table made up of a collection of Series.
# Series is a one-dimensional array, while a DataFrame is a multi-dimensional array.

data = {
    'ages': [14, 18, 24, 42],
    'heights': [165, 180, 176, 184]
}

print(data)
print(data['ages'])
print(data['ages'][1])
print(data['ages'], data['heights'])

df = pd.DataFrame(data)

print(df)
print(df['ages'])
print(df['ages'][1])
print(df.iloc[1])
print(df[['ages', 'heights']])

df = pd.DataFrame(data, index=['James', 'Bob', 'Amy', 'Dave'])

print(df)
print(type(df))

print(df['ages'])
print(type(df['ages']))

print(df['ages'][1])
print(type(df['ages'][1]))

print(df['ages']['Bob'])
print(type(df['ages']['Bob']))

print(df.loc['Bob'])
print(type(df.loc['Bob']))

print(df.iloc[1])
print(type(df.iloc[1]))

print(df.iloc[:3])
print(type(df.iloc[:3]))

print(df.iloc[1:3])
print(type(df.iloc[1:3]))

print(df.iloc[1:])
print(type(df.iloc[1:]))

print(df.iloc[-2:])
print(type(df.iloc[-2:]))

print(df[['ages', 'heights']])
print(type(df[['ages', 'heights']]))

print(df[df['ages'] > 18])
print(type(df[df['ages'] > 18]))

print(df[(df['ages'] > 18) & (df['heights'] > 180)])
print(type(df[(df['ages'] > 18) & (df['heights'] > 180)]))

print(df[(df['ages'] > 18) | (df['heights'] > 180)])
print(type(df[(df['ages'] > 18) | (df['heights'] > 180)]))
