import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df = pd.read_csv('csv_files/titanic.csv')
print(df.head())

df['male'] = df['Sex'] == 'male'
print(df.head())

print(df['Fare'].values)

arr = df[['Pclass', 'Fare', 'Age']].values
print(arr)
print(arr.shape)
print(arr[0])
print(arr[:, 2])
print(arr[0, 1])

mask = arr[:, 2] < 18
print(arr[mask])
print(mask.sum())
print(arr[arr[:, 2] < 18])

plt.scatter(df['Age'], df['Fare'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.savefig('./plots/age_fare.png')
plt.show()

plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.savefig('./plots/age_fare_class.png')
plt.show()

plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.plot([0, 80], [85, 5])
plt.savefig('./plots/age_fare_class_line.png')
plt.show()
