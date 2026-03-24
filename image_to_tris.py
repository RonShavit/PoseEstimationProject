import cv2
import trimap_beta as tm

def sample_pixels(image_path, margin):
    # Load image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found or path is incorrect")

    # Convert to grayscale (black & white)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    samples = []

    # Iterate with step = margin
    for y in range(0, height, margin):
        for x in range(0, width, margin):
            r = int(gray[y, x])/255.0  # grayscale value (0–255)
            samples.append((x, y, r))

    return samples


def build_triangles(samples, width, height, margin):
    """
    samples: list of (x, y, r) in row-major order
    width, height: original image dimensions
    margin: sampling step

    returns: list of triangles (each triangle is a tuple of 3 indices)
    """

    # Number of sampled points per row
    cols = (width + margin - 1) // margin
    rows = (height + margin - 1) // margin

    def idx(x_idx, y_idx):
        """Convert grid coordinates to flat index"""
        return y_idx * cols + x_idx

    triangles = []

    for y in range(rows - 1):
        for x in range(cols - 1):
            # Current point
            i0 = idx(x, y)

            # Right neighbor (a+m, b)
            i1 = idx(x + 1, y)

            # Bottom neighbor (a, b+m)
            i2 = idx(x, y + 1)

            # Bottom-right (a+m, b+m)
            i3 = idx(x + 1, y + 1)

            # Triangle 1: (a,b), (a+m,b), (a,b+m)
            triangles.append((i0, i1, i2))

            # Triangle 2: (a+m,b), (a+m,b+m), (a,b+m)
            triangles.append((i1, i3, i2))

    return triangles

samp = sample_pixels("map_1.jpg", 10)
image = cv2.imread("map_1.jpg")
height, width, _ = image.shape
tris = build_triangles(samp, width, height, 10)  # Replace 640 and 480 with actual image dimensions
with open("test2.tri", 'w') as f:
    for i, (x, y, r) in enumerate(samp):
        f.write(f"{x},{y},{r}\n")
    
    for i in tris:
        f.write(f"v{i[0]},v{i[1]},v{i[2]}\n")