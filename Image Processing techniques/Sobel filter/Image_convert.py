import numpy as np
from PIL import Image

# 1. Load both arrays
Gx = np.loadtxt("Gx_aligned.txt")
Gy = np.loadtxt("Gy_aligned.txt")

# 2. Compute gradient magnitude: sqrt(Gx² + Gy²)
mag = np.sqrt(Gx**2 + Gy**2)

# Whats happening ^ is Gx^2 is squaring each values of each element inside the Gx array and then adding them to the squared value array of Gy and then

# 3. Normalize to 0–255
max_val = mag.max() if mag.max() != 0 else 1
mag_norm = (mag / max_val) * 255.0

# 4. Convert to uint8
mag_img = mag_norm.astype(np.uint8)


# 5. Save as image
img = Image.fromarray(mag_img)
img.save("edges.png")

