import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv('csv_files/titanic.csv')

df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

model = LogisticRegression()

scores = cross_val_score(model, X, y, cv=5)
print('Cross-Validation Accuracy Scores:', scores)
print(scores.mean())
