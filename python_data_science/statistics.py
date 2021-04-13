import numpy as np

# In Python, lists are used to store data.
# NumPy provides an array structure for performing operations with data.
# NumPy arrays are faster and more compact than lists.
# NumPy arrays are homogeneous.
# They  can contain only a single data type, while lists can contain multiple different types of data.
num_list = [18, 24, 67, 55, 42, 14, 19, 26, 33]
np_array = np.array(num_list)
print(np_array[3])
print(np_array.mean())
print(np_array.max(initial=None))
print(np_array.min(initial=None))
print(np_array.std())
print(np_array.var())

players = [180, 172, 178, 185, 190, 195, 192, 200, 210, 190]
np_array = np.array(players)
std = np_array.std()
mean = np_array.mean()
count = 0
for player in players:
    if mean + std >= player >= mean - std:
        count += 1
print(count)

# NumPy arrays are often called ndarrays, which stands for 'N-dimensional array'
# They can have multiple dimensions.
ndarray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])
print(ndarray[1][2])
print(ndarray.ndim)
print(ndarray.size)
print(ndarray.shape)

x = np.array([2, 1, 3])
x = np.append(x, 4)
x = np.delete(x, 0)
x = np.sort(x)
print(x)

print(np.arange(2, 20, 3))
