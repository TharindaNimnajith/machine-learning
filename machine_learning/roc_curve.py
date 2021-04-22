import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

sensitivity_score = recall_score


def specificity_score(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]


df = pd.read_csv('csv_files/titanic.csv')

df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy: {0:.5f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: {0:.5f}'.format(precision_score(y_test, y_pred)))
print('Recall: {0:.5f}'.format(recall_score(y_test, y_pred)))
print('F1 Score: {0:.5f}'.format(f1_score(y_test, y_pred)))

print('Sensitivity:', sensitivity_score(y_test, y_pred))
print('Specificity:', specificity_score(y_test, y_pred))

print(model.predict_proba(X_test))
print(model.predict_proba(X_test)[:, 1])

y_pred = model.predict_proba(X_test)[:, 1] > 0.75

print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))

y_pred_proba = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.savefig('plots/roc_curve.png')
plt.show()

model1 = LogisticRegression()
model1.fit(X_train, y_train)
y_pred_proba1 = model1.predict_proba(X_test)
print('Model 1 AUC Score:', roc_auc_score(y_test, y_pred_proba1[:, 1]))

model2 = LogisticRegression()
model2.fit(X_train[:, 0:2], y_train)
y_pred_proba2 = model2.predict_proba(X_test[:, 0:2])
print('Model 2 AUC Score:', roc_auc_score(y_test, y_pred_proba2[:, 1]))
