import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('csv_files/titanic.csv')

df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

print('Whole dataset:', X.shape, y.shape)
print('Training set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# building the model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluating the model
# print('Accuracy:', model.score(X_test, y_test))
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y, y_pred))
print('Precision:', precision_score(y, y_pred))
print('Recall:', recall_score(y, y_pred))
print('F1 Score:', f1_score(y, y_pred))
print(confusion_matrix(y, y_pred))
