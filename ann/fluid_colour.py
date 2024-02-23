import cv2
import numpy as np
import noise
import random

width = 1920
height = 1080

# Load the image
input = cv2.imread('output_b&w.png', cv2.IMREAD_GRAYSCALE)
grid_y, grid_x = input.shape
image = cv2.resize(input, dsize=(width, height), interpolation=cv2.INTER_NEAREST)

# Define the dimensions of each segment
segment_width = width // grid_x
segment_height = height // grid_y

##### Define color maps (BGR) #####

#beige and greys
# transitions = 128
# color_map = [
#     [177, 199, 210],
#     [52, 49, 40],
#     [178, 189, 185],
#     [43, 61, 65]
# ]

#grey and pink pink
# transitions = 128
# color_map = [
#     [177, 199, 210],
#     [52, 49, 40],
#     [178, 189, 185],
#     [215, 207, 230]
# ]

#grey and lighter pink
# transitions = 128
# color_map = [
#     [177, 199, 210],
#     [52, 49, 40],
#     [178, 189, 185],
#     [193, 148, 225]
# ]

# transitions = 128
# color_map = [
#     [155, 149, 142],
#     [134, 117, 114],
#     [191, 140, 134],
#     [44, 84, 67]
# ]

#orange backgriound with white and black
# transitions = 32
# color_map = [
#     [3, 51, 182],
#     [175, 201, 220],
#     [28, 32, 38]
# ]

#wheat field
# transitions = 64
# color_map = [
#     [171, 198, 218],
#     [120, 77, 65],
#     [55, 112, 212],
#     [70, 196, 243],
#     [69, 63, 59]
# ]

#flame with bit of blue spice
# transitions = 128
# color_map = [
#     [36, 36, 45],
#     [105, 202, 225],
#     [54, 59, 164],
#     [191, 124, 63]
# ]


# Generate Perlin noise for fluid motion
def generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity, seed):
    perlin_noise = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            perlin_noise[y, x] = noise.pnoise2(x * scale, 
                                               y * scale,
                                               octaves=octaves,
                                               persistence=persistence,
                                               lacunarity=lacunarity,
                                               repeatx=width,
                                               repeaty=height,
                                               base=seed)
    return perlin_noise


# Define a function to create fluid-like motion
def fluid_motion(image, perlin_noise):
    # Create a copy of the image
    new_image = np.copy(image)
    rows, cols = image.shape
    
    # Iterate over each pixel and apply displacement based on Perlin noise
    for y in range(rows):
        for x in range(cols):
            # Calculate displacement using Perlin noise
            displacement_x = int(perlin_noise[y, x] * 150)  # Adjust the multiplier for displacement strength
            displacement_y = int(perlin_noise[y, x] * 150)
            
            # Apply displacement to the pixel position
            new_x = max(0, min(cols - 1, x + displacement_x))
            new_y = max(0, min(rows - 1, y + displacement_y))
            
            # Set the pixel value at the new position
            new_image[y, x] = image[new_y, new_x]
    
    return new_image

def colour(image, num_transitions):
    rgb_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Calculate the transition range for each color segment
    transition_range = 255 / (len(color_map) - 1)

    # Map grayscale values to colors with fade effect
    for i in range(len(color_map) - 1):
        start_color = color_map[i]
        end_color = color_map[i + 1]
        lower_bound = int(i * transition_range)
        upper_bound = int((i + 1) * transition_range)
        for j in range(lower_bound, upper_bound):
            weight = (j - lower_bound) / (upper_bound - lower_bound)
            interpolated_color = [
                int(start_color[k] * (1 - weight) + end_color[k] * weight)
                for k in range(3)
            ]
            rgb_img[np.where(image == j)] = interpolated_color
    return rgb_img

# Function to continuously update and display the image
def update_image():
    num_transitions = transitions
    for i in range(20):
        name = f"fluid_colour/fluid_motion_{i:02d}.png"

        # Add randomness to the seed
        random.seed()  # Seed with current system time
        perlin_seed = random.randint(0, 1000)  # Generate a random seed
        print(i, ": ", perlin_seed)
        perlin_noise = generate_perlin_noise(1920, 1080, 0.05, 6, 0.5, 2.0, perlin_seed)

        moved_image = fluid_motion(image, perlin_noise)

        coloured_image = colour(moved_image, num_transitions)
        print(num_transitions)
        num_transitions = num_transitions - 16
        
        cv2.imwrite(name, coloured_image)

# Call the function to start displaying the image
update_image()

# Release resources
cv2.destroyAllWindows()