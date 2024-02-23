import cv2
import os
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
            displacement_x = int(perlin_noise[y, x] * 200)  # Adjust the multiplier for displacement strength
            displacement_y = int(perlin_noise[y, x] * 200)
            
            # Apply displacement to the pixel position
            new_x = max(0, min(cols - 1, x + displacement_x))
            new_y = max(0, min(rows - 1, y + displacement_y))
            
            # Set the pixel value at the new position
            new_image[y, x] = image[new_y, new_x]
    
    return new_image

# Function to continuously update and display the image
def update_image():
    for i in range(20):
        name = f"fluid/fluid_motion_{i:02d}.png"

        # Add randomness to the seed
        random.seed()  # Seed with current system time
        perlin_seed = random.randint(0, 1000)  # Generate a random seed
        print(i, ": ", perlin_seed)
        perlin_noise = generate_perlin_noise(1920, 1080, 0.05, 6, 0.5, 2.0, perlin_seed)

        moved_image = fluid_motion(image, perlin_noise)
        
        cv2.imwrite(name, moved_image)

# Call the function to start displaying the image
update_image()

# Release resources
cv2.destroyAllWindows()