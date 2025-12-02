import numpy as np
from PIL import Image

img = Image.open("Human.png").convert("L")
    # What ".convert("L")" does is convert the RGB image into a black and white one by following this formula "0.299 R + 0.587 G + 0.114 B" where the Red values are multiplied by 0.299 and Green is multiplied by 0.587 etc...
img = np.array(img, dtype=np.int32)

# What dtype=np.int32 does is tell numpy to expect the datatype to be a 32 bit signed integer, a signed integer is an integer that can store both negative and positive values.

# Sobel kernels
Kx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

Ky = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

h, w = img.shape

# Prepare output arrays
Gx = np.zeros((h, w), dtype=np.int32)
Gy = np.zeros((h, w), dtype=np.int32)

# np.zeros returns an array of integers of the same height and width as the original image (h,w) in line 23

# Apply Sobel convolution
for i in range(1, h - 1):
    for j in range(1, w - 1):
        block = img[i-1:i+2, j-1:j+2]

        gx_val = np.sum(Kx * block)
        gy_val = np.sum(Ky * block)

        Gx[i, j] = gx_val
        Gy[i, j] = gy_val

# Print results
np.savetxt("Gx_aligned.txt", Gx, "%d")
np.savetxt("Gy_aligned.txt", Gy, "%d")
