import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 1000)
# plt.plot(x, np.sin(x))
# plt.plot(x, np.cos(x))
plt.plot(x, np.sin(x), color='k')
plt.plot(x, np.cos(x), color='r', linestyle='--')
plt.plot(x, np.sin(x), 'k:', label='sin(x)')
plt.plot(x, np.cos(x), 'r--', label='cos(x)')
plt.legend()
plt.savefig('plots/sin_cos.png')
# plt.draw()
# plt.pause(1)
# plt.show()
# plt.clf()
