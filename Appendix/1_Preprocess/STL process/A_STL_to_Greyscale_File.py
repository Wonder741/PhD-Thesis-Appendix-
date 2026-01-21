# This script converts a 3D STL file into a grayscale PNG image with customizable resolution.
# 
# Usage:
# - Input: Provide the path to the STL file, the desired output directory and output resolution.
# - Note: The output resolution must be an integer multiple of the default resolution (320) for proper scaling.
# - Output: The script generates a PNG file in the output directory with the same name as the input STL file but with a .png extension.
# 
# How it works:
# 1. The STL file is loaded and processed using the trimesh library.
# 2. Each triangle in the STL is sampled to generate points, and their height values are converted into grayscale values.
# 3. The grayscale image is centered and scaled to fit the specified resolution.
# 4. Black (unfilled) pixels in the image are filled based on neighboring pixel values.
# 5. The resulting image is saved as a PNG file in the specified output directory.
# 
# To run the script:
# - Or import the process_stl_to_png(input_file, output_path) function into another script and call it directly.

import os
import cv2
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import generic_filter

# Function to convert height to greyscale value
def height_to_greyscale(height):
    if height <= -20:
        return 0
    if height >= 0:
        return 255
    return int(255 + (height / 20) * 255)

# Function to sample multiple points within a triangle
def sample_points_in_triangle(v0, v1, v2, num_samples=20):
    samples = []
    for _ in range(num_samples):
        # Generate random barycentric coordinates
        r1 = np.random.rand()
        r2 = np.random.rand()
        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = r2 * sqrt_r1
        w = 1 - u - v
        # Calculate the sample point
        point = u * v0 + v * v1 + w * v2
        samples.append(point)
    return samples

def fill_black_pixels(image):
    def fill_pixel(values):
        # Check if the center pixel is black (value 0)
        if values[4] == 0:
            # Count the number of non-zero surrounding pixels
            non_zero_count = np.count_nonzero(values[np.arange(9) != 4])
            # If 6, 7, or 8 of the surrounding pixels are non-zero, fill the pixel
            if non_zero_count >= 7:
                non_zero_values = values[values > 0]
                return int(np.mean(non_zero_values))
        return values[4]  # Return the original value if conditions aren't met

    # Apply the filter to each pixel in the image
    filled_image = generic_filter(image, fill_pixel, size=3, mode='constant', cval=0)
    return filled_image

# Function to process STL file and create greyscale image
def process_stl_to_png(input_file, output_file, output_resolution):
    default_resolution = 320
    scaling_factor = output_resolution / default_resolution
    mesh = trimesh.load_mesh(input_file)

    # Get the bounding box of the mesh
    min_bound, max_bound = mesh.bounds
    object_width_pixels = max_bound[0] - min_bound[0]
    object_depth_pixels = max_bound[1] - min_bound[1]

    # Center offset to place the object at the image's center
    x_offset = round((default_resolution - object_width_pixels) / 2)
    y_offset = round((default_resolution - object_depth_pixels) / 2)

    # Create a 2D grid for the image, with 0 as the background
    image = np.zeros((default_resolution, default_resolution), dtype=np.uint8)

    # Translate and scale the mesh vertices
    vertices = mesh.vertices.copy()
    vertices[:, 0] = (vertices[:, 0] - min_bound[0]) + x_offset
    vertices[:, 1] = (vertices[:, 1] - min_bound[1]) + y_offset

    # Iterate over each face and sample multiple points
    # Adding a progress bar for face processing
    for face in tqdm(mesh.faces, desc="Processing STL", unit="faces"):
        v0, v1, v2 = vertices[face]
        sampled_points = sample_points_in_triangle(v0, v1, v2, num_samples=20)

        for point in sampled_points:
            x, y, z = point
            grey_value = height_to_greyscale(z)

            px = int(np.clip((default_resolution - 1 - y), 0, (default_resolution - 1)))
            py = int(np.clip(x, 0, (default_resolution - 1)))
            image[px, py] = grey_value

    # Apply the post-processing step to fill black pixels
    image = fill_black_pixels(image)
    center_offset = 0

    # Scale the image to match output_resolution if it differs from default_resolution
    if output_resolution != default_resolution:
        new_image = np.zeros((output_resolution, output_resolution), dtype=np.uint8)

        # Compute new size
        new_size = (int(image.shape[1] * scaling_factor), int(image.shape[0] * scaling_factor))

        # Resize the image using interpolation (cubic for better quality)
        scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

        # Ensure the scaled image is large enough to be cropped
        center_offset_x = (scaled_image.shape[1] - output_resolution) // 2
        center_offset_y = (scaled_image.shape[0] - output_resolution) // 2

        # Handle cases where the scaled image is too small
        if center_offset_x < 0 or center_offset_y < 0:
            # Pad the image instead of slicing
            pad_x = max(0, -center_offset_x)
            pad_y = max(0, -center_offset_y)
            scaled_image = np.pad(scaled_image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')

            # Recalculate center offsets after padding
            center_offset_x = max(0, (scaled_image.shape[1] - output_resolution) // 2)
            center_offset_y = max(0, (scaled_image.shape[0] - output_resolution) // 2)

        # Crop to output resolution
        new_image[:output_resolution, :output_resolution] = scaled_image[
            center_offset_y:center_offset_y + output_resolution,
            center_offset_x:center_offset_x + output_resolution
        ]

        image = new_image

    # Scale the object dimensions
    scaled_width = round(object_width_pixels) * scaling_factor
    scaled_height = round(object_depth_pixels) * scaling_factor

    # Scale the boundaries
    scaled_min_x = (x_offset - 1) * scaling_factor - center_offset
    scaled_max_x = (round(max_bound[0] - min_bound[0]) + x_offset - 1) * scaling_factor - center_offset
    scaled_min_y = (y_offset - 1) * scaling_factor - center_offset
    scaled_max_y = (round(max_bound[1] - min_bound[1]) + y_offset - 1) * scaling_factor - center_offset

    # Save the image
    plt.imsave(output_file, image, cmap='gray', vmin=0, vmax=255)
    print(f"Converted stl to greyscale png")

    # Return the scaled dimensions and scaled boundaries
    scaled_boundaries = [scaled_min_x, scaled_max_x, scaled_min_y, scaled_max_y]
    return [round(scaled_width, 2), round(scaled_height, 2)] + [int(round(val + 0.5)) for val in scaled_boundaries]

# Main function to call the processing function
def main(input_file, output_path, output_resolution):
    object_size = process_stl_to_png(input_file, output_path, output_resolution)
    print(object_size)

# Example usage when called from another script
if __name__ == "__main__":
    file_name = "left_01wangchongguang.stl"
    output_resolution = 320
    input_folder = r"D:\Data\Complete project\4 Data processing Python\1 STL to Greyscale\3D STL"
    output_folder = r"D:\Data\Complete project\4 Data processing Python\1 STL to Greyscale\Greyscale"
    input_file = os.path.join(input_folder, file_name)
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    main(input_file, output_folder, output_resolution)
