import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./csv/titanic.csv')
plt.scatter(df['Fare'], df['Age'], c=df['Survived'])
plt.xlabel('Fare')
plt.ylabel('Age')
plt.plot([30, 110], [0, 80])
plt.savefig('./plots/titanic_fare_age.png')
plt.show()

df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

print(X)
print(y)

print(X.shape)
print(y.shape)
