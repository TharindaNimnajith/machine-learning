import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 1000)
y = np.sin(x)
fig = plt.figure()
axis = plt.axes()
axis.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('function sin(x)')
plt.xlim(2, 8)
plt.ylim(0, 1)
plt.savefig('plots/sin_modified.png')
# plt.draw()
# plt.pause(1)
# plt.show()
# plt.clf()
