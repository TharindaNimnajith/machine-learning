import pandas as pd

# df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df = pd.read_csv('./titanic.csv')
print(df.head())

df['male'] = df['Sex'] == 'male'
print(df.head())

print(df['Fare'].values)

arr = df[['Pclass', 'Fare', 'Age']].values
print(arr)
print(arr.shape)
