# This script aligns a grayscale points image with a corresponding object image for registration.
#
# Usage:
# - Input:
#   1. Provide the path to the object image file (e.g., an insole shape in grayscale).
#   2. Provide the path to the points image file (e.g., a scanned 3D foot image in grayscale).
#   3. Specify the directories for debug output and processed results.
# - Output:
#   - Generates a processed points image aligned with the object image.
#   - Creates a debug overlay image showing the alignment results.
#   - Outputs alignment results (e.g., translation, optimal angle) in a CSV file.
#
# How it works:
# 1. Loads the object and points images in grayscale.
# 2. Aligns the center of the points image to the center of the object image.
# 3. Rotates the points image to maximize overlap with the object mask.
# 4. Fine-tunes alignment by testing small translations (bias) in the x and y directions.
# 5. Saves the processed image, debug overlay image, and results to the specified directories.
#
# To run the script:
# - Execute the script directly with predefined file paths and directories.
# - Or import the `main(object_image_file, points_image_path, debug_dir, processed_dir)` function into another script and call it directly.

import cv2
import numpy as np
import os
import csv
import pandas as pd

# Use the minimum frame method to locate the point image center, and move the center to the image center (320, 320)
def align_centers(points_mask):
    frame_center = calculate_object_center(points_mask)

    # Calculate the center of the object mask
    # Get the dimensions of the image
    height, width = points_mask.shape[:2]
    image_center = [width // 2, height // 2]

    # Calculate the translation needed to move the frame center to the object center
    translation = np.array(image_center) - np.array(frame_center)

    # Apply the translation to the points mask
    rows, cols = points_mask.shape
    M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    aligned_points_mask = cv2.warpAffine(points_mask, M, (cols, rows))

    return aligned_points_mask, translation, image_center

#Calculate the center of an object in a grayscale image.
def calculate_object_center(image):
    # Get the coordinates of all non-zero pixels
    non_zero_coords = np.argwhere(image > 0)

    if non_zero_coords.size == 0:
        raise ValueError("The image does not contain any non-zero pixels.")

    # Extract x and y coordinates
    y_coords, x_coords = non_zero_coords[:, 0], non_zero_coords[:, 1]

    # Get bounding box coordinates
    xmin, xmax = x_coords.min(), x_coords.max()
    ymin, ymax = y_coords.min(), y_coords.max()

    # Calculate the center of the object
    x_center = xmin + (xmax - xmin) / 2
    y_center = ymin + (ymax - ymin) / 2

    center = [x_center, y_center]
    return center

# Rotated the aligned point image based on image center, find the optimal angle that have the maximum overlap with the object mask
def test_rotation(object_mask, aligned_points_mask, rotation_center):
    rows, cols = aligned_points_mask.shape
    max_overlap = 0
    best_angle = 0
    optimal_rotate_image = None
    a = np.sum((aligned_points_mask) > 0)

    for angle in range(0, 360, 1):  # Rotate from 0 to 180 degrees in steps of 10
        M = cv2.getRotationMatrix2D(rotation_center, angle, 1)
        rotated_points = cv2.warpAffine(aligned_points_mask, M, (cols, rows), flags=cv2.INTER_NEAREST)
        _, rotated_points = cv2.threshold(rotated_points, 127, 255, cv2.THRESH_BINARY)
        overlap = np.sum((object_mask & rotated_points) > 0)
        #print(a, overlap, angle)

        if max_overlap < overlap:
            max_overlap = overlap
            best_angle = angle
            optimal_rotate_image = rotated_points
            #print(a, overlap, angle)
    return optimal_rotate_image, max_overlap, best_angle

# Align the roated point image in a short range x(-bias_range ,bias_range) and y(-bias_range, bias_range) to further optimal the image alignment
def test_bias(object_mask, points_mask, bias_range=20):
    rows, cols = points_mask.shape
    max_overlap = 0
    optimal_bias = (0, 0)
    optimal_moved_image = None
    a = np.sum((points_mask) > 0)
    #print(a)

    # Test different biases
    for dx in range(-bias_range, bias_range + 1):
        for dy in range(-bias_range, bias_range + 1):
            # Apply the bias
            M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
            translated_points = cv2.warpAffine(points_mask, M_translate, (cols, rows))

            # Calculate the overlap
            overlap = np.sum((object_mask & translated_points) > 0)

            # Update the maximum overlap and optimal bias
            if overlap > max_overlap:
                max_overlap = overlap
                optimal_bias = (dx, dy)
                optimal_moved_image = translated_points

    return optimal_moved_image, max_overlap, optimal_bias

def update_csv_by_filename(file_path, filename, tx, ty, optimal_angle):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            # Create the file and write the header
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Filename', 'tx', 'ty', 'optimal_angle'])
            print(f"File created with header: {file_path}")
        
        # Read the existing data
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            rows = [row for row in reader]
        
        # Ensure the header has the correct columns
        column_count = len(header)
        if header != ['Filename', 'tx', 'ty', 'optimal_angle']:
            raise ValueError("The CSV file has an incorrect header.")

        # Check if the filename exists and update or append the data
        updated = False
        for row in rows:
            if row[0] == filename:
                # Extend the row to match the number of header columns
                while len(row) < column_count:
                    row.append('')
                
                # Update the relevant columns
                row[1] = tx
                row[2] = ty
                row[3] = optimal_angle
                updated = True
        
        # Append the row if no match is found
        if not updated:
            rows.append([filename, tx, ty, optimal_angle])
            print(f"New row added for {filename}.")

        # Write the updated data back to the file
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Data updated successfully in {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def align_loop(object_image_file, points_image_file, debug_dir, processed_dir):
    # Set output path
    directory_path, file_name = os.path.split(points_image_file)
    debug_image_path = os.path.join(debug_dir, file_name)
    processed_points_image_path = os.path.join(processed_dir, file_name)
    csv_file_path = os.path.join(processed_dir, "result.csv")

    # Load the object mask
    object_image = cv2.imread(object_image_file, cv2.IMREAD_GRAYSCALE)
    object_edges = cv2.Canny(object_image, 50, 200)
    object_mask = cv2.threshold(object_image, 50, 255, cv2.THRESH_BINARY)[1]
    object_mask_edge = cv2.threshold(object_edges, 50, 255, cv2.THRESH_BINARY)[1]

    # Load and preprocess the points image
    points_image = cv2.imread(points_image_file, cv2.IMREAD_GRAYSCALE)
    points_mask = cv2.threshold(points_image, 235, 255, cv2.THRESH_BINARY)[1]

    # Align centers, rotate, and find optimal bias
    aligned_points_mask, translation, centroid = align_centers(points_mask)

    # Initialize optimal angle, translation and overlap
    optimal_angle = 0
    optimal_translation = translation
    current_overlap = 0
    max_overlap = 10

    while(current_overlap < max_overlap):
        current_overlap = max_overlap
        optimal_rotate_image, max_overlap, best_angle = test_rotation(object_mask, aligned_points_mask, centroid)
        
        if 181 <= best_angle <= 359:
            best_angle = best_angle - 360
        

        if (current_overlap < max_overlap):
            optimal_angle += best_angle
            rows, cols = points_image.shape
            M_rotate = cv2.getRotationMatrix2D(centroid, best_angle, 1)  # Rotation
            rotated_points_image = cv2.warpAffine(points_image, M_rotate, (cols, rows))

        aligned_points_mask, max_overlap, optimal_bias = test_bias(object_mask, optimal_rotate_image)
        if (current_overlap < max_overlap):
            optimal_translation[0] += optimal_bias[0]
            optimal_translation[1] += optimal_bias[1]
            M_translate = np.float32([[1, 0, optimal_bias[0]], [0, 1, optimal_bias[1]]])  # Bias
            aligned_points_image = cv2.warpAffine(rotated_points_image, M_translate, (cols, rows))

        centroid = calculate_object_center(aligned_points_mask)

    processed_points_image = aligned_points_image
    # Overlay the object mask and the modified points mask for debugging
    debug_image = cv2.cvtColor(object_mask_edge, cv2.COLOR_GRAY2BGR)
    debug_image[:, :, 1] = np.maximum(debug_image[:, :, 1], optimal_rotate_image)

    # Save the debug image
    cv2.imwrite(debug_image_path, debug_image)
    # Save the processed points image
    cv2.imwrite(processed_points_image_path, processed_points_image)

    # Convert the processed image to a DataFrame
    df = pd.DataFrame(processed_points_image)
    # Save the DataFrame to a CSV file
    csv_path = os.path.join(processed_dir, f"{file_name}.csv")
    df.to_csv(csv_path, index=False, header=False)

    print(f"Foot shape aligned to insole")

     # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image File Name', 'Optimal Angle', 'dx', 'dy'])
        # Write the results to the CSV file
        csv_writer.writerow([file_name, optimal_angle, optimal_translation[0], optimal_translation[1]])
    
    return processed_points_image

# Main function to call the processing function
def main(object_image_file, points_image_path, debug_dir, processed_dir):
    processed_points_image = align_loop(object_image_file, points_image_path, debug_dir, processed_dir)

# Example usage when called from another script
if __name__ == "__main__":
    sides = ["left", "right"]
    length = 270
    id_folder = "37weijie"
    size = str(int(length / 10 + 1))

    object_image_path = r"D:\A\B_Propcess_data\Insoles"
    points_image_path = r"D:\A\B_Propcess_data\3DScan\2Greyscale"
    debug_base_dir = rf"D:\A\B_Propcess_data\3DScan\3Debug\{id_folder}"
    processed_base = r"D:\A\B_Propcess_data\3DScan\4Processed"

    csv_file_name = os.path.join(processed_base, "processed_results.csv")

    for side in sides:
        insole_name = side + "_" + size + ".png"
        id_name = side + "_" + id_folder + ".png"

        debug_dir = os.path.join(debug_base_dir, side)
        processed_dir = os.path.join(processed_base, id_folder, side)

        object_image_file = os.path.join(object_image_path, side, insole_name)
        points_image_file = os.path.join(points_image_path, id_folder, id_name)

        # Create output directories if they don't exist
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        # Call the main function for processing
        processed_points_image = main(object_image_file, points_image_file, debug_dir, processed_dir)
