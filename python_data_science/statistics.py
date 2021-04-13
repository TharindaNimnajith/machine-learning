import numpy as np

num_list = [18, 24, 67, 55, 42, 14, 19, 26, 33]

np_array = np.array(num_list)

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
