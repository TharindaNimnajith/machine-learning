import numpy as np

heights = [189, 170, 189, 163, 183, 171, 185, 168, 173, 183, 173, 173, 175, 178, 183, 193, 178, 173, 174, 183, 183, 180,
           168, 180, 170, 178, 182, 180, 183, 178, 182, 188, 175, 179, 183, 193, 182, 183, 177, 185, 188, 188, 182, 185,
           191]

cnt = 0
for height in heights:
    if height > 188:
        cnt += 1
print(cnt)

heights_arr = np.array(heights)
print(heights_arr)
print((heights_arr > 188).sum())
print(heights_arr.dtype)
print(heights_arr.shape)
print(len(heights_arr))

heights_arr[3] = 165
print(heights_arr)

heights_arr = heights_arr.reshape((3, 15))
print(heights_arr)
print(heights_arr[1, 2])
print(heights_arr[1][2])
print(heights_arr[0, 0:3])
print(heights_arr[0, :3])
print(heights_arr[:, 3])
print(heights_arr[1, 0:3])
print(heights_arr[2, -3:])
print(heights_arr[-1, 3])

heights_arr[1, 1] = 176
print(heights_arr)

heights_arr[0, :] = 180
print(heights_arr)

heights_arr[:2, :2] = 0
print(heights_arr)

heights_arr[:, 0] = [100, 200, 300]
print(heights_arr)

new = np.array([[50, 150, 250], [25, 125, 225], [75, 175, 275]])
print(new)
print(new.shape)
print(heights_arr)
print(heights_arr.shape)
print(heights_arr[:, 12:])
print(heights_arr[:, 12:].shape)
heights_arr[:, 12:] = new
print(heights_arr)

heights = [189, 170, 189, 163, 183, 171, 185, 168, 173, 183, 173, 173, 175, 178, 183, 193, 178, 173, 174, 183, 183, 180,
           168, 180, 170, 178, 182, 180, 183, 178, 182, 188, 175, 179, 183, 193, 182, 183, 177, 185, 188, 188, 182, 185,
           191]

ages = [57, 61, 57, 57, 58, 57, 61, 54, 68, 51, 49, 64, 50, 48, 65, 52, 56, 46, 54, 49, 51, 47, 55, 55, 54, 42, 51, 56,
        55, 51, 54, 51, 60, 62, 43, 55, 56, 61, 52, 69, 64, 46, 54, 47, 70]

print(heights)
print(ages)

heights_arr = np.array(heights)
ages_arr = np.array(ages)

print(heights_arr)
print(ages_arr)

heights_arr = heights_arr.reshape((45, 1))
ages_arr = ages_arr.reshape((45, 1))

print(heights_arr)
print(ages_arr)

height_age_arr = np.hstack((heights_arr, ages_arr))
print(height_age_arr)
print(height_age_arr.shape)
print(height_age_arr[:3, ])

heights_arr = np.array(heights)
ages_arr = np.array(ages)

heights_arr = heights_arr.reshape((1, 45))
ages_arr = ages_arr.reshape((1, 45))

print(heights_arr)
print(ages_arr)

height_age_arr = np.vstack((heights_arr, ages_arr))
print(height_age_arr)
print(height_age_arr.shape)
print(height_age_arr[:, :3])

heights_arr = np.array(heights)
ages_arr = np.array(ages)

heights_arr = heights_arr.reshape((45, 1))
ages_arr = ages_arr.reshape((45, 1))

height_age_arr = np.concatenate((heights_arr, ages_arr), axis=1)
print(height_age_arr)
print(height_age_arr.shape)
print(height_age_arr[:3, :])

heights_arr = np.array(heights)
ages_arr = np.array(ages)

heights_arr = heights_arr.reshape((1, 45))
ages_arr = ages_arr.reshape((1, 45))

height_age_arr = np.concatenate((heights_arr, ages_arr), axis=0)
print(height_age_arr)
print(height_age_arr.shape)
print(height_age_arr[:, :3])

heights_arr = np.array(heights)
ages_arr = np.array(ages)

heights_arr = heights_arr.reshape((45, 1))
ages_arr = ages_arr.reshape((-1, 1))

height_age_arr = np.hstack((heights_arr, ages_arr))
print(height_age_arr[:, 0] * 0.0328084)

print(height_age_arr.sum())
print(height_age_arr.sum(axis=0))

print(height_age_arr.min(initial=None))
print(height_age_arr.min(initial=None, axis=0))

print(height_age_arr.max(initial=None))
print(height_age_arr.max(initial=None, axis=0))

print(height_age_arr.mean())
print(height_age_arr.mean(axis=0))

print(height_age_arr[:, 1])
print(height_age_arr[:, 1] < 55)
print(height_age_arr[:, 1] == 51)
print((height_age_arr[:, 1] == 51).sum())

mask = height_age_arr[:, 1] == 51
print(mask.sum())

other = height_age_arr[mask,]
print(other)
print(other.shape)

mask = (height_age_arr[:, 0] >= 182) & (height_age_arr[:, 1] <= 50)
print(height_age_arr[mask,])

test_input = '2 2\n1.5 1\n2 2.9'
n = [float(x) for x in test_input.split()]
arr = np.array(n[2:])
arr = arr.reshape((int(n[0]), int(n[1])))
print(arr.mean(axis=1))

# import numpy as np
#
# contents = []
#
# while True:
#     try:
#         contents.append(input().strip().split())
#     except EOFError:
#         break
#
# print(np.array(contents[1:]).astype(np.float).mean(axis=1))
