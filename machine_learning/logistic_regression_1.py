import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('csv_files/titanic.csv')

X = df[['Fare', 'Age']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(X, y)

# save the model to disk
filename = './models/logistic_regression_1.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model)

print(model.coef_, model.intercept_)  # [[ 0.01615949 -0.01549065]] [-0.51037152]
print(model.coef_[0, 0])  # 0.01615949
print(model.coef_[0, 1])  # -0.01549065
print(model.intercept_[0])  # -0.51037152

y = np.linspace(0, 80, 100)

x = (model.coef_[0, 1] * -1 * y + model.intercept_[0] * -1) / model.coef_[0, 0]
# x = (0.01549065 * y + 0.51037152) / 0.01615949

plt.plot(x, y, '-b', label='Logistic Regression - Survival by Age and Fare')
plt.xlabel('Fare')
plt.ylabel('Age')
plt.scatter(df['Fare'], df['Age'], c=df['Survived'])
plt.grid()
plt.savefig('./plots/logistic_regression_1.png')
plt.show()
