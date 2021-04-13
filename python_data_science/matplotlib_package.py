import matplotlib.pyplot as plt
import pandas as pd

# Matplotlib is a library used to create graphs, charts, and figures. It also provides functions to customize your
# figures by changing the colors and labels.
help(plt)
print(dir(plt))
print(plt)

s = pd.Series([18, 42, 9, 32, 81, 64, 3])
s.plot(kind='bar')
plt.savefig('plot.png')
