import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

help(mpl)
print(dir(mpl))
print(mpl)

plt.style.use('ggplot')
print(plt)
print(plt.style)

fig = plt.figure()
ax = plt.axes()
plt.savefig('plots/fig.png')
# plt.draw()
# plt.pause(1)
# plt.show()
# plt.clf()
