import matplotlib as mpl

help(mpl)
print(dir(mpl))
print(mpl)

import matplotlib.pyplot as plt

plt.style.use('ggplot')
print(plt)
print(plt.style)

fig = plt.figure()
ax = plt.axes()
plt.savefig('plots/fig.png')
# plt.show()

plt.figure()
plt.ion()
ax1 = plt.subplot(211)
plt.title('test', fontsize=8)
plt.xlim(-1700, 1700)
plt.ylabel('x-axis')
plt.xlabel('y-axis')
plt.grid()
plt.savefig('plots/stackoverflow.png')
# plt.show()
plt.clf()
