import numpy as np
from PIL import Image

img = np.array(Image.open("input.png").convert("L"), dtype=np.int32)

kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])

h, w = img.shape
out = np.zeros((h, w), dtype=np.int32)

for i in range(1, h-1):
    for j in range(1, w-1):
        block = img[i-1:i+2, j-1:j+2]
        out[i, j] = np.sum(kernel * block)

# Convert to viewable image
out_abs = np.abs(out)
out_norm = (out_abs / out_abs.max() * 255).astype(np.uint8)

Image.fromarray(out_norm).save("laplacian_edges.png")
