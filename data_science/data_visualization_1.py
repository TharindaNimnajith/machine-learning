import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 1000)
y = np.sin(x)
fig = plt.figure()
axis = plt.axes()
axis.plot(x, y)
plt.savefig('plots/sin_basic.png')
# plt.draw()
# plt.pause(1)
# plt.show()
# plt.clf()
