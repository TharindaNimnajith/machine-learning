import numpy as np
import pandas as pd

data = '3 4 5 3 4 4 nan'
lst = [float(x) if x != 'nan' else np.NaN for x in data.split()]
arr = np.array(lst)
df = pd.Series(arr)
df.fillna((df.mean().round(1)), inplace=True)
print(df)
