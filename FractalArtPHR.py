import numpy as np
from PIL import Image, ImageOps
import cv2
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Function to generate concentric circles pattern
def generate_concentric_circles_pattern(grid_size, circle_frequency):
    X, Y = create_grid(grid_size)
    Z = np.sqrt(X**2 + Y**2)
    Z = np.sin(circle_frequency * Z)
    return Z

# Create a grid of coordinates
def create_grid(grid_size):
    x = np.linspace(-2 * np.pi, 2 * np.pi, grid_size)
    y = np.linspace(-2 * np.pi, 2 * np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    return X, Y

# Normalize the pattern
def normalize_pattern(Z):
    Z_min, Z_max = Z.min(), Z.max()
    Z_normalized = (Z - Z_min) / (Z_max - Z_min)
    return Z_normalized

# Create a gradient colormap with random colors
def create_random_colormap(num_colors=8):
    colors = ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(num_colors)]
    cmap = LinearSegmentedColormap.from_list("random_cmap", colors)
    return cmap

# Apply the gradient colormap to the pattern
def apply_colormap(Z_normalized, cmap):
    norm = plt.Normalize(Z_normalized.min(), Z_normalized.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colored_image = sm.to_rgba(Z_normalized, bytes=True)[:, :, :3]  # Remove alpha channel
    return colored_image

# Create the kaleidoscope pattern
def create_kaleidoscope_pattern(colored_image, tile_size):
    image_pil = Image.fromarray(colored_image)
    height, width = colored_image.shape[:2]
    
    kaleidoscope_image = Image.new("RGB", (width, height))

    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            quarter_image = image_pil.crop((j, i, j + tile_size, i + tile_size)).resize((tile_size, tile_size))
            kaleidoscope_image.paste(quarter_image, (j, i))
            kaleidoscope_image.paste(ImageOps.mirror(quarter_image), (width - j - tile_size, i))
            kaleidoscope_image.paste(ImageOps.flip(quarter_image), (j, height - i - tile_size))
            kaleidoscope_image.paste(ImageOps.mirror(ImageOps.flip(quarter_image)), (width - j - tile_size, height - i - tile_size))
    
    return kaleidoscope_image

# Parameters
grid_size = 800  # Change grid size to control the overall size of the pattern

# Manually set parameters for concentric circles illusion
circle_frequency = 10  # Adjust the frequency of the concentric circles

# Generate concentric circles pattern
Z = generate_concentric_circles_pattern(grid_size, circle_frequency)

# Normalize the pattern
Z_normalized = normalize_pattern(Z)

# Create random gradient colormap
cmap = create_random_colormap()

# Apply colormap to the pattern
colored_image = apply_colormap(Z_normalized, cmap)

# Choose tile size for kaleidoscope pattern
tile_size = 200  # Change tile size to control the detail level of the kaleidoscope pattern

# Create kaleidoscope image
kaleidoscope_image = create_kaleidoscope_pattern(colored_image, tile_size)

# Save the image using Pillow
kaleidoscope_image.save("ConcentricCirclesIllusionArtPiece.png")

# Optionally, display the image using OpenCV
image_cv = cv2.cvtColor(np.array(kaleidoscope_image), cv2.COLOR_RGB2BGR)
cv2.imshow("Concentric Circles Illusion Art Piece", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()