import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.array(['dog', 'cat', 'cat', 'dog', 'dog'])
y_pred = np.array(['dog', 'cat', 'cat', 'cat', 'dog'])

print(confusion_matrix(y_true, y_pred, labels=['cat', 'dog']))
