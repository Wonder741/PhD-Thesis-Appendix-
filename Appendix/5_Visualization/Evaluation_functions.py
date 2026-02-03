import os
import numpy as np
from PIL import Image
factor_a = 3.14 * (0.005 / 2) ** 2 * 1000 #0.019625 sensor area
factor_1 = 46    # force parameter
factor_2 = 35    # force parameter

def adc_force(arr):
    arr_f = np.zeros_like(arr, dtype=np.float32)

    # condition 1: arr > 184
    mask_high = arr > 184
    arr_f[mask_high] = (arr[mask_high] - 44) / factor_1

    # condition 2: 0 < arr <= 184
    mask_mid = (arr > 0) & (arr <= 184)
    arr_f[mask_mid] = arr[mask_mid] / factor_2
    return arr_f

def count_grayscale_images(folder_path):
    """
    Returns the number of images in folder_path whose mode is 'L' (grayscale).
    """
    count = 0
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        try:
            with Image.open(path) as img:
                if img.mode == 'L':
                    count += 1
        except (IOError, FileNotFoundError):
            # not an image, or can’t be opened
            continue
    return count

import os
import csv
from PIL import Image
import numpy as np
import re

def write_csv(csv_file, id_folder, insole_file_name, side, top_max, top_fname, bottom_max, bottom_fname, contact_area):
    file_exists = os.path.isfile(csv_file)
    # Append result
    factor_area = 0.25
    contact_area = contact_area * factor_area
    top_max = top_max 
    bottom_max = bottom_max 
    with open(csv_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['id_folder', 'insole_file_name', 'side', 'top_max', 'top_fname', 'bottom_max', 'bottom_fname', 'contact area'])
        writer.writerow([id_folder, insole_file_name, side, top_max, top_fname, bottom_max, bottom_fname, contact_area])
    #print(f"Appended values for {insole_file_name} to {csv_file}")

def max_pixel(folder_path, output_path):
    """
    Processes all grayscale PNG images in the given folder.

    Input path expected format:
      .../{id_folder}/{insole_file_name}
    - id_folder is the second-lowest folder name
    - insole_file_name is the lowest folder name, formatted: {number}{char}_{name}_{number1}_{number2}
      where {char} is 'L' or 'R'.

    The function finds the maximum pixel value over all images in this folder and
    appends a line to max_value.csv in the parent directory of id_folder.

    CSV columns: id_folder, insole_file_name, side, max_value.

    Parameters:
        folder_path (str): Path to the folder containing grayscale PNGs.
    """
    # Normalize path and split
    norm_path = os.path.normpath(folder_path)
    parts = norm_path.split(os.sep)
    if len(parts) < 2:
        print(f"Error: path too short to extract id_folder and insole_file_name: {folder_path}")
        return

    # Extract identifiers
    id_folder = parts[-2]
    insole_file_name = parts[-1]

    # Determine side from insole_file_name
    match = re.match(r"^(\d+)([LR])_", insole_file_name)
    if not match:
        print(f"Skipping folder with unexpected name: {insole_file_name}")
        return
    char = match.group(2)
    side = "left" if char == "L" else "right"

    # Verify folder exists
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # Scan grayscale images and find top‐half and bottom‐half maxima
    top_max = -1
    bottom_max = -1
    top_fname = ""
    bottom_fname = ""
    found = False

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith('.png'):
            continue

        img_path = os.path.join(folder_path, fname)
        try:
            img = Image.open(img_path)
            if img.mode != 'L':
                # skip non-grayscale
                continue

            arr = np.array(img, dtype=np.uint8)
            h, w = arr.shape
            mid = h // 2
            arr_f = adc_force(arr) // factor_a
            # Compute max in each half
            curr_top_max = arr_f[:mid, :].max() if mid > 0 else -1
            curr_bottom_max = arr_f[mid:, :].max() if h - mid > 0 else -1

            # Update running maxima
            if curr_top_max > top_max:
                top_max = curr_top_max
                top_fname = fname
            if curr_bottom_max > bottom_max:
                bottom_max = curr_bottom_max
                bottom_fname = fname

            found = True

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if not found:
        print(f"No grayscale PNG images found in {folder_path}")
    #else:
        #print(f"Top‐half max value: {top_max}")
        #print(f"Bottom‐half max value: {bottom_max}")

    # Prepare CSV path in parent of id_folder
    csv_dir = os.path.join(output_path, id_folder, insole_file_name)
    csv_file = os.path.join(csv_dir, id_folder + '_report.csv')
    os.makedirs(csv_dir, exist_ok=True)
    #print(f"Processed folder {folder_path}: max pixel value = {overall_max}")
    return csv_file, id_folder, insole_file_name, side, top_max, top_fname, bottom_max, bottom_fname
# Example usage function max:
# max_pixel(folder_path, output_path)

import os
import csv
from PIL import Image
import numpy as np

def process_contact_area(folder_path, output_path):
    """
    Process a folder of grayscale insole PNG images to compute and save the contact area.

    For the given `folder_path`, this function:
      1. Extracts an `id_folder` (parent folder name) and `insole_file_name` (folder name).
      2. Determines the foot `side` ("left" or "right") from the filename pattern.
      3. Loads all grayscale PNGs in the folder, computes their pixel‐wise average.
      4. Thresholds the average image to produce a binary “contact area” mask:
         pixels > 1.5 → 255, else 0.
      5. Saves this mask as `<output_path>/<id_folder>/<insole_file_name>/<id_folder>_contact.png`.
      6. Counts the non‐zero pixels in the mask (`max_area`).
      7. Prepares a report CSV path `<output_path>/<id_folder>/<insole_file_name>/<id_folder>_report.csv`.

    Parameters:
        folder_path (str):
            Path to the folder containing the insole’s grayscale PNG images.
            Folder name must match the regex `r"^(\d+)([LR])_"` to extract side.
        output_path (str):
            Base directory under which the contact image and report CSV will be saved.

    Returns:
        tuple:
            csv_file (str): Full path where the report CSV should be written.
            id_folder (str): Identifier extracted from the parent folder name.
            insole_file_name (str): Name of the insole folder.
            side (str): Either "left" or "right", parsed from the filename.
            max_area (int): Number of pixels in the contact area mask (non‐zero count).

    Notes:
      - Skips any non‐grayscale or non‐PNG files.
      - If the folder path is invalid or no images are found, the function prints an error and returns None.
    """
    # Normalize path and split
    norm_path = os.path.normpath(folder_path)
    parts = norm_path.split(os.sep)
    if len(parts) < 2:
        print(f"Error: path too short to extract id_folder and insole_file_name: {folder_path}")
        return

    # Extract identifiers
    id_folder = parts[-2]
    insole_file_name = parts[-1]

    # Determine side from insole_file_name
    match = re.match(r"^(\d+)([LR])_", insole_file_name)
    if not match:
        print(f"Skipping folder with unexpected name: {insole_file_name}")
        return
    char = match.group(2)
    side = "left" if char == "L" else "right"

    # Verify folder exists
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    images = []
    found_image = False
    # Loop over all PNG files in the folder
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith('.png'):
            continue
        found_image = True
        img_path = os.path.join(folder_path, fname)
        try:
            img = Image.open(img_path)
            # Ensure image is in grayscale mode ('L')
            if img.mode != 'L':
                # skip non-grayscale
                continue
            img_array = np.array(img, dtype=np.float32)
            images.append(img_array)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if not found_image:
        print(f"No grayscale PNG images found in {folder_path}")
        return

    # Compute pixel-wise average over the image sequence (assumes all images have the same dimensions)
    stacked = np.stack(images, axis=0)
    avg_image = np.mean(stacked, axis=0)

    # Create a binary image: non-zero pixels become 255, zeros remain 0
    binary_image = np.where(avg_image > 1.5, 255, 0).astype(np.uint8)

    # Define the destination folder for saving the output image
    img_dir = os.path.join(output_path, id_folder, insole_file_name)
    img_file = os.path.join(img_dir, f"{id_folder}_contact.png")
    os.makedirs(img_dir, exist_ok=True)

    # Convert to PIL Image
    pil_img = Image.fromarray(binary_image)

    # Compute new size (width ×10, height ×10)
    new_width  = pil_img.width  * 5
    new_height = pil_img.height * 5

    # Resize using nearest-neighbor to preserve the binary look
    resized_img = pil_img.resize((new_width, new_height), Image.NEAREST)
    
    # Save the resized image
    resized_img.save(img_file)
    #print(f"Saved contact area image to {output_image_path}")

    # Count the number of non-zero pixels in the binary image (contact area)
    max_area = int(np.count_nonzero(binary_image))

    # Prepare CSV path in parent of id_folder
    csv_dir = os.path.join(output_path, id_folder, insole_file_name)
    csv_file = os.path.join(csv_dir, id_folder + '_report.csv')
    os.makedirs(csv_dir, exist_ok=True)
    return csv_file, id_folder, insole_file_name, side, max_area
    #print(f"Appended max_area data: id={id_}, number={number}, side={side}, max_area={max_area}")

# Example usage function contact area
#process_contact_area(folder_path)

import os
import re
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt

def time_integral_image(folder_path, output_path, start_frame, end_frame):
    """
    Compute the time-integral (sum) of grayscale PNG frames in `folder_path` whose filenames
    end with _{frame}.png, where start_frame <= frame <= end_frame. Clips, scales, and
    generates both greyscale and jet-colormap heatmaps, and writes out CSV and images.

    Parameters:
        folder_path (str):   Directory containing grayscale PNGs.
        output_path (str):   Base path where results will be saved.
        start_frame (int):   Minimum frame number to include (inclusive).
        end_frame (int):     Maximum frame number to include (inclusive).
    """
    # Normalize and split the folder path
    norm_path = os.path.normpath(folder_path)
    parts = norm_path.split(os.sep)
    if len(parts) < 2:
        print(f"Error: path too short to extract id_folder and insole_file_name: {folder_path}")
        return

    id_folder = parts[-2]
    insole_file_name = parts[-1]

    # Determine side
    m_side = re.match(r"^(\d+)([LR])_", insole_file_name)
    if not m_side:
        print(f"Skipping folder with unexpected name: {insole_file_name}")
        return
    side = "left" if m_side.group(2) == 'L' else 'right'

    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # Regex to extract frame number
    frame_pattern = re.compile(r"_(\d+)\.png$", re.IGNORECASE)

    sum_matrix = None
    image_count = 0

    # Loop through files and accumulate only in-range frames
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith('.png'):
            continue
        m = frame_pattern.search(fname)
        if not m:
            continue
        frame = int(m.group(1))
        if frame < start_frame or frame > end_frame:
            continue

        img_path = os.path.join(folder_path, fname)
        try:
            img = Image.open(img_path)
            if img.mode != 'L':
                continue  # skip non-grayscale
            arr = np.array(img, dtype=np.int64)
            arr_p = adc_force(arr) // factor_a
            if sum_matrix is None:
                sum_matrix = arr_p
            else:
                sum_matrix += arr_p
            image_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if image_count == 0:
        print(f"No valid PNG frames in [{start_frame}–{end_frame}] found in {folder_path}")
        return

    second_matrix = sum_matrix // ((end_frame - start_frame))
    # Clip and scale
    clipped = np.clip(second_matrix, 0, np.max(second_matrix))
    scaled = (clipped * (255.0 / np.max(second_matrix))).astype(np.uint8)
    #scaled = np.where(scaled < 1.5, 0, scaled)

    # Greyscale and jet images
    image_gray = Image.fromarray(scaled, mode='L')

    normed = scaled.astype(np.float32) / 255.0
    colored = cm.jet(normed)
    rgb = (colored[..., :3] * 255).astype(np.uint8)
    mask = (scaled == 0)
    rgb[mask] = 0
    image_jet = Image.fromarray(rgb, mode='RGB')

    # Prepare output directories
    img_dir = os.path.join(output_path, id_folder, insole_file_name)
    os.makedirs(img_dir, exist_ok=True)

    # Save outputs
    out_csv = os.path.join(img_dir, f"{id_folder}_{side}_time_integral_{start_frame}-{end_frame}.csv")
    np.savetxt(out_csv, second_matrix, delimiter=",", fmt="%d")

    out_gray = os.path.join(img_dir, f"{id_folder}_{side}_time_integral_{start_frame}-{end_frame}_g.png")
    out_jet  = os.path.join(img_dir, f"{id_folder}_{side}_time_integral_{start_frame}-{end_frame}_h.png")

    # Compute new size (width ×10, height ×10)
    new_width  = image_gray.width  * 5
    new_height = image_gray.height * 5

    # Resize using nearest-neighbor to preserve the binary look
    resized_gray = image_gray.resize((new_width, new_height), Image.NEAREST)
    resized_jet = image_jet.resize((new_width, new_height), Image.NEAREST)

    resized_gray.save(out_gray)
    # Convert resized_jet back to an array
    jet_arr = np.array(resized_jet)

    # Compute the original data range
    data_max = np.max(second_matrix) 
    data_min = 0

    # Create a norm object mapping [data_min, data_max] → [0,1]
    norm = plt.Normalize(vmin=data_min, vmax=data_max)

    # Plot
    fig, ax = plt.subplots(figsize=(jet_arr.shape[1] / 100, jet_arr.shape[0] / 100), dpi=100)
    # Show the image using the same colormap
    im = ax.imshow(jet_arr, cmap='jet', norm=norm, origin='upper')
    ax.axis('off')

    # Add colorbar at right, with ticks at [data_max, data_min]
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_ticks([data_max, data_min])
    cbar.set_ticklabels([f"{data_max:.0f}", f"{data_min:.0f}"])
    cbar.ax.invert_yaxis()  # so that the top of the bar is data_max

    # Save the combined image+legend
    fig.savefig(out_jet, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    #resized_jet.save(out_jet)

    #print(f"Processed {image_count} frames in [{start_frame}–{end_frame}].")
    #print(f"Saved CSV to {out_csv}")
    #print(f"Saved greyscale map to {out_gray}")
    #print(f"Saved heatmap to {out_jet}")

import os
import re
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt

def max_intensity_image(folder_path, output_path, start_frame, end_frame):
    """
    Compute the per-pixel maximum value over a sequence of grayscale PNG frames
    in `folder_path` whose filenames end with _{frame}.png, where
    start_frame <= frame <= end_frame. Generates:
      - a greyscale image of the max values,
      - a false-color (jet) image with zeros shown as black,
      - a CSV file storing the max-value matrix.

    Parameters:
        folder_path (str): Directory containing grayscale PNGs.
        output_path (str): Base path where results will be saved.
        start_frame (int): Minimum frame number (inclusive).
        end_frame (int): Maximum frame number (inclusive).
    """
    # Normalize and split the folder path
    norm_path = os.path.normpath(folder_path)
    parts = norm_path.split(os.sep)
    if len(parts) < 2:
        print(f"Error: path too short: {folder_path}")
        return

    id_folder = parts[-2]
    insole_file_name = parts[-1]

    # Determine side L/R
    m_side = re.match(r"^(\d+)([LR])_", insole_file_name)
    if not m_side:
        print(f"Skipping folder with unexpected name: {insole_file_name}")
        return
    side = 'left' if m_side.group(2) == 'L' else 'right'

    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # Frame extraction regex
    frame_pattern = re.compile(r"_(\d+)\.png$", re.IGNORECASE)

    max_matrix = None
    image_count = 0

    # Iterate through frames
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith('.png'):
            continue
        m = frame_pattern.search(fname)
        if not m:
            continue
        frame = int(m.group(1))
        if frame < start_frame or frame > end_frame:
            continue

        img_path = os.path.join(folder_path, fname)
        try:
            img = Image.open(img_path)
            if img.mode != 'L':
                continue  # skip non-grayscale
            arr = np.array(img, dtype=np.int64)
            arr_p = adc_force(arr) / factor_a

            if max_matrix is None:
                max_matrix = arr_p
            else:
                max_matrix = np.maximum(max_matrix, arr_p)
            image_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if image_count == 0:
        print(f"No valid frames in range [{start_frame}-{end_frame}] found.")
        return

    # Prepare output directory
    img_dir = os.path.join(output_path, id_folder, insole_file_name)
    os.makedirs(img_dir, exist_ok=True)

    # Save CSV of max values
    out_csv = os.path.join(img_dir, f"{id_folder}_{side}_maxintensity_{start_frame}-{end_frame}.csv")
    np.savetxt(out_csv, max_matrix, delimiter=",", fmt="%d")

    # Scale to 0-255 for image
    clipped = np.clip(max_matrix, 0, np.max(max_matrix))
    scaled = (clipped * (255.0 / np.max(clipped))).astype(np.uint8)
    # Threshold tiny values to zero
    scaled = np.where(scaled < 1, 0, scaled)

    # Greyscale image
    img_gray = Image.fromarray(scaled, mode='L')
    out_gray = os.path.join(img_dir, f"{id_folder}_{side}_maxintensity_{start_frame}-{end_frame}_gray.png")
    
    # Compute new size (width ×10, height ×10)
    new_width  = img_gray.width  * 5
    new_height = img_gray.height * 5

    # Resize using nearest-neighbor to preserve the binary look
    resized_gray = img_gray.resize((new_width, new_height), Image.NEAREST)
    resized_gray.save(out_gray)

    # Jet colormap image
    normed = scaled.astype(np.float32) / 255.0
    colored = cm.jet(normed)
    rgb = (colored[..., :3] * 255).astype(np.uint8)
    # Set zeros to black
    mask = (scaled == 0)
    rgb[mask] = 0
    img_jet = Image.fromarray(rgb, mode='RGB')
    out_jet = os.path.join(img_dir, f"{id_folder}_{side}_maxintensity_{start_frame}-{end_frame}_jet.png")

    resized_jet = img_jet.resize((new_width, new_height), Image.NEAREST)
    # Convert resized_jet back to an array
    jet_arr = np.array(resized_jet)

    # Compute the original data range
    data_max = np.max(max_matrix) 
    data_min = 0

    # Create a norm object mapping [data_min, data_max] → [0,1]
    norm = plt.Normalize(vmin=data_min, vmax=data_max)

    # Plot
    fig, ax = plt.subplots(figsize=(jet_arr.shape[1] / 100, jet_arr.shape[0] / 100), dpi=100)
    # Show the image using the same colormap
    im = ax.imshow(jet_arr, cmap='jet', norm=norm, origin='upper')
    ax.axis('off')

    # Add colorbar at right, with ticks at [data_max, data_min]
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_ticks([data_max, data_min])
    cbar.set_ticklabels([f"{data_max:.0f}", f"{data_min:.0f}"])
    cbar.ax.invert_yaxis()  # so that the top of the bar is data_max

    # Save the combined image+legend
    fig.savefig(out_jet, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    #print(f"Processed {image_count} frames.")
    #print(f"Saved matrix CSV to {out_csv}")
    #print(f"Saved greyscale image to {out_gray}")
    #print(f"Saved jet image to {out_jet}")


import os
import re
import csv
import numpy as np
from PIL import Image

def centre_pressure_image(folder_path, output_path, start_frame, end_frame):
    """
    Compute the center‐of‐pressure (center of gravity) for each grayscale PNG
    in `folder_path` whose filename ends with _{frame}.png, where
    start_frame <= frame <= end_frame. Marks each CoP on a blank image and
    writes out a CSV of (x,y) coords.

    Parameters:
        folder_path (str):      Path to the folder of grayscale PNGs.
        output_path (str):      Base path where results will be saved.
        start_frame (int):      Minimum frame number to include (inclusive).
        end_frame (int):        Maximum frame number to include (inclusive).
    """
    # extract IDs from path
    norm_path = os.path.normpath(folder_path)
    parts = norm_path.split(os.sep)
    if len(parts) < 2:
        print(f"Error: path too short to extract id_folder and insole_file_name: {folder_path}")
        return
    id_folder, insole_file_name = parts[-2], parts[-1]

    # determine side
    m = re.match(r"^(\d+)([LR])_", insole_file_name)
    if not m:
        print(f"Skipping folder with unexpected name: {insole_file_name}")
        return
    side = "left" if m.group(2) == "L" else "right"

    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # prep regex to pull frame from filename
    frame_pat = re.compile(r"_(\d+)\.png$", re.IGNORECASE)
    centroids = []
    image_size = None

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".png"):
            continue

        fm = frame_pat.search(fname)
        if not fm:
            continue
        frame = int(fm.group(1))
        if frame < start_frame or frame > end_frame:
            continue

        img_path = os.path.join(folder_path, fname)
        try:
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')
            arr = np.array(img, dtype=np.uint16) * 2
            arr = np.clip(arr, 0, 255).astype(np.uint8)

            # first‐image sets size
            if image_size is None:
                image_size = arr.shape

            nz_rows, nz_cols = np.nonzero(arr)
            if nz_rows.size == 0:
                continue

            weights = arr[nz_rows, nz_cols].astype(np.float64)
            total_w = weights.sum()
            if total_w < 250:
                continue

            # weighted centroid = center of gravity
            r_cog = int(round((nz_rows * weights).sum() / total_w))
            c_cog = int(round((nz_cols * weights).sum() / total_w))
            centroids.append((r_cog, c_cog))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if image_size is None:
        print("No valid frames found in the specified range.")
        return

    # build a blank image and plot CoP points
    new_img = np.zeros(image_size, dtype=np.uint8)
    for r, c in centroids:
        if 0 <= r < image_size[0] and 0 <= c < image_size[1]:
            new_img[r, c] = 255

    # save image
    img_dir = os.path.join(output_path, id_folder, insole_file_name)
    os.makedirs(img_dir, exist_ok=True)
    out_img = os.path.join(img_dir, f"{id_folder}_{side}_COP_{start_frame}-{end_frame}.png")

    # Convert to PIL Image
    pil_img = Image.fromarray(new_img)

    # Compute new size (width ×10, height ×10)
    new_width  = pil_img.width  * 5
    new_height = pil_img.height * 5

    # Resize using nearest-neighbor to preserve the binary look
    resized_img = pil_img.resize((new_width, new_height), Image.NEAREST)

    # Save the resized image
    resized_img.save(out_img)
    #Image.fromarray(new_img).save(out_img)

    # save CSV of (x, y)
    csv_path = os.path.join(img_dir, f"{id_folder}_{side}_COP_{start_frame}-{end_frame}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])
        for r, c in centroids:
            writer.writerow([c, r])

    #print(f"Saved CoP image to {out_img}")
    #print(f"Saved CoP coords to {csv_path}")

def find_threshold_indices(lst, threshold=400):
    """
    1) From the start of lst, find the first index `s` where lst[s] < threshold.
    2) In the sublist lst[s:], find the index of its maximum value → max_idx.
    3) Around max_idx, find:
         - `a`: the first index to the left where value < threshold
         - `b`: the first index to the right where value < threshold
    4) If `a` is never found, set a = 1; if `b` is never found, set b = len(lst).
    Returns: (a, b)
    """
    if not lst:
        return 1, len(lst)

    # 1) find s
    s = None
    for i, v in enumerate(lst):
        if v < threshold:
            s = i
            break
    if s is None:
        # never dropped below threshold
        return 1, len(lst)

    # 2) find max in lst[s:]
    sub = lst[s:]
    max_val = max(sub)
    max_idx = s + sub.index(max_val)

    # 3) search left of max_idx for first < threshold
    a = None
    for i in range(max_idx - 1, -1, -1):
        if lst[i] < threshold:
            a = i
            break

    #    search right of max_idx for first < threshold
    b = None
    for i in range(max_idx + 1, len(lst)):
        if lst[i] < threshold:
            b = i
            break

    # 4) apply defaults if not found
    if a is None:
        a = 1
    if b is None:
        b = len(lst)

    return a - 5, b + 5

def find_threshold_indices_self(lst):
    """
    1. Calculate threshold = max(lst) * 0.5
    2. Find s1 and s2:
       - s1: first index from start where lst[i] < threshold
       - s2: first index from end where lst[i] < threshold
    3. Between s1 and s2, find the index of the maximum value → max_idx.
    4. Around max_idx, find:
         - `a`: the first index to the left where value < threshold
         - `b`: the first index to the right where value < threshold
    5. If `a` is never found, set a = 1; if `b` is never found, set b = len(lst).
    Returns: (a, b)
    """
    n = len(lst)
    if n == 0:
        return 0, 0

    # 1. threshold
    threshold = max(lst) * 0.5

    # 2. find s1 (from start)
    s1 = next((i for i, v in enumerate(lst) if v < threshold), None)
    # find s2 (from end)
    rev = next((i for i, v in enumerate(reversed(lst)) if v < threshold), None)
    s2 = n - 1 - rev if rev is not None else None

    # if either not found, return defaults
    if s1 is None or s2 is None:
        return 1, n

    # 3. find max index between s1 and s2
    segment = lst[s1:s2 + 1]
    max_val = max(segment)
    max_idx = s1 + segment.index(max_val)

    # 4. find a (left)
    a = next((i for i in range(max_idx - 1, -1, -1) if lst[i] < threshold), None)
    # find b (right)
    b = next((i for i in range(max_idx + 1, n) if lst[i] < threshold), None)

    # 5. defaults
    if a is None:
        a = 1
    if b is None:
        b = n

    return a - 10, b + 10

import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def ground_reaction_force(folder_path, output_path, start_frame, end_frame):
    """
    Compute and plot ground reaction force (pixel‐sum) for grayscale PNG frames
    in `folder_path` whose filenames end with _{frame}.png and satisfy
    start_frame <= frame <= end_frame.

    Parameters:
        folder_path (str):   Directory containing .png images.
        output_path (str):   Base path where results will be saved.
        start_frame (int):   Minimum frame number to include (inclusive).
        end_frame (int):     Maximum frame number to include (inclusive).
    """
    # Normalize path and split
    norm_path = os.path.normpath(folder_path)
    parts = norm_path.split(os.sep)
    if len(parts) < 2:
        print(f"Error: path too short to extract id_folder and insole_file_name: {folder_path}")
        return

    # Extract identifiers
    id_folder = parts[-2]
    insole_file_name = parts[-1]

    # Determine side from insole_file_name
    match = re.match(r"^(\d+)([LR])_", insole_file_name)
    if not match:
        print(f"Skipping folder with unexpected name: {insole_file_name}")
        return
    side = "left" if match.group(2) == "L" else "right"

    # Verify folder exists
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    data_points = []  # list of (frame, sum_of_pixels)
    frame_pattern = re.compile(r"_(\d+)\.png$", re.IGNORECASE)

    # Loop over all PNG files in the folder
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".png"):
            continue

        m = frame_pattern.search(filename)
        if not m:
            print(f"Filename does not match expected pattern: {filename}")
            continue

        frame = int(m.group(1))
        if frame < start_frame or frame > end_frame:
            continue  # skip outside the requested range

        image_path = os.path.join(folder_path, filename)
        try:
            # Load as grayscale
            img = Image.open(image_path)
            if img.mode != 'L':
                img = img.convert('L')
            arr = np.array(img, dtype=np.uint8)

            # sum pixels
            arr_f = adc_force(arr)

            pixel_sum = int(arr_f.sum())
            data_points.append((frame, pixel_sum))

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    if not data_points:
        print("No valid frames found in the specified range.")
        return

    # Sort and unpack
    data_points.sort(key=lambda x: x[0])
    frames    = [pt[0] for pt in data_points]
    sums      = [pt[1] for pt in data_points]
    times_sec = [f / 50 for f in frames]   # if each frame = 1/50 s
    forces_N  = [s for s in sums]     # calibration: sum/25 → N

    a, b = find_threshold_indices_self(forces_N)

    # Plot
    plt.figure(figsize=(20, 4))
    plt.plot(times_sec, forces_N, marker='o', linestyle='-')
    plt.xlabel("Time (s)")
    plt.ylabel("Ground Reaction Force (N)")
    plt.title(f"Ground Force Plot for {id_folder}_{side}")

    # Save plot
    img_dir = os.path.join(output_path, id_folder, insole_file_name)
    os.makedirs(img_dir, exist_ok=True)
    plot_path = os.path.join(img_dir, f"{id_folder}_{side}_GRF_{start_frame}-{end_frame}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    # Save raw data CSV (frame, pixel_sum)
    csv_path = os.path.join(img_dir, f"{id_folder}_{side}_GRF_{start_frame}-{end_frame}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "pixel_sum"])
        for fr, sm in data_points:
            writer.writerow([fr, sm])

    #print(f"Saved GRF plot to {plot_path}")
    #print(f"Saved GRF data to {csv_path}")
    return a, b

import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_heatmap(input_folder, output_path, start_frame, end_frame, z_scale=1.5, dot_size=0.5):
    """
    Load and visualize a range of 64×64 grayscale PNG images as a 3D heatmap scatter.

    Only processes files in `input_folder` matching the pattern `..._{frame}.png` where
    `start_frame <= frame <= end_frame`. Each image is mapped through the jet colormap
    and stacked along the Z axis according to its frame number.

    Parameters:
        input_folder (str): Path containing 64×64 grayscale PNGs named ..._{frame}.png
        output_folder (str): Directory to save the resulting plot image.
        start_frame (int):   Minimum frame number to include (inclusive).
        end_frame (int):     Maximum frame number to include (inclusive).
        z_scale (float):     Vertical spacing multiplier for successive frames.
        dot_size (float):    Size of each scatter point.
    """
    # Normalize path and split
    norm_path = os.path.normpath(input_folder)
    parts = norm_path.split(os.sep)
    if len(parts) < 2:
        print(f"Error: path too short to extract id_folder and insole_file_name: {input_folder}")
        return

    # Extract identifiers
    id_folder = parts[-2]
    insole_file_name = parts[-1]

    # ensure output exists
    img_dir = os.path.join(output_path, id_folder, insole_file_name)
    os.makedirs(img_dir, exist_ok=True)

    # regex to extract frame number
    pattern = re.compile(r"_(\d+)\.png$", re.IGNORECASE)
    frames = []

    # collect and filter frames
    for fname in os.listdir(input_folder):
        match = pattern.search(fname)
        if not match:
            continue
        frame = int(match.group(1))
        if frame < start_frame or frame > end_frame:
            continue
        frames.append((frame, os.path.join(input_folder, fname)))

    if not frames:
        print(f"No valid PNGs in [{start_frame}–{end_frame}] found in {input_folder}")
        return

    # sort by frame number
    frames.sort(key=lambda x: x[0])

    # read first image in grayscale to get dimensions
    sample_path = frames[0][1]
    sample_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    if sample_img is None:
        print(f"Failed to load sample image: {sample_path}")
        return
    h, w = sample_img.shape

    # create coordinate grid
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    # setup 3D plot
    fig = plt.figure(figsize=(10, 30))
    ax = fig.add_subplot(111, projection='3d')

    # iterate through each frame
    for frame_num, img_path in frames:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # normalize pixel values to [0,1]
        img_norm = img.astype(np.float32) / 127.0

        # map to jet colormap and set zero pixels transparent
        colors = cm.jet(img_norm)
        colors[img_norm == 0, 3] = 0

        # flatten grids and colors
        Xf = X.ravel()
        Yf = Y.ravel()
        Zf = np.full_like(Xf, frame_num * z_scale, dtype=np.float32)
        Cf = colors.reshape(-1, 4)

        ax.scatter(Xf, Yf, Zf, c=Cf, s=dot_size, marker='.')

    # labels and title
    ax.set_xlabel('X (width)')
    ax.set_ylabel('Y (height)')
    ax.set_zlabel('Frame number')
    ax.set_title(os.path.basename(os.path.normpath(input_folder)))

    # maintain aspect ratio
    z0 = frames[0][0] * z_scale
    z1 = frames[-1][0] * z_scale
    dz = z1 - z0 if z1 != z0 else 1
    ax.set_box_aspect((w, h, dz * 4))

    # format z-axis ticks back to frame numbers
    ax.zaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, pos: f"{val / z_scale:.0f}")
    )
    ax.view_init(elev=30, azim=20)

    # save figure
    out_name = f"{id_folder}_{start_frame}-{end_frame}_3d.png"
    out_path = os.path.join(img_dir, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    #print(f"Saved 3D heatmap scatter for frames {start_frame}–{end_frame} to {out_path}")

if __name__ == "__main__":
    folder_path = input("Enter prediction directory path: "); print("Valid path." if os.path.exists(folder_path) else "Invalid path.")
    # Determine project root (one level above this script's directory)
    output_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_path, "4Result")
    start_frame = 1
    end_frame = 1200
    csv_file, id_folder, insole_file_name, side, overall_max = max_pixel(folder_path, output_path)
    csv_file, id_folder, insole_file_name, side, contact_area = process_contact_area(folder_path, output_path)
    time_integral_image(folder_path, output_path, start_frame, end_frame)
    write_csv(csv_file, id_folder, insole_file_name, side, overall_max, contact_area)
    centre_pressure_image(folder_path, output_path, start_frame, end_frame)
    ground_reaction_force(folder_path, output_path, start_frame, end_frame)
    #plot_3d_heatmap(folder_path, output_path, start_frame, end_frame)

import os
import re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def resize_image(folder_path, output_path, start_frame, end_frame, start_sample, end_sample):
    """
    Instead of computing GRF, this function:
      - Resizes grayscale images 10× between start_frame and end_frame
      - Applies a false-color 'jet' colormap
      - Saves colored images to output_path/Resize with the same filenames
      - Quantizes the resized grayscale into 8 steps and saves to output_path/Step with the same filenames
    """

    # Normalize path and split
    norm_path = os.path.normpath(folder_path)
    parts = norm_path.split(os.sep)
    if len(parts) < 2:
        print(f"Error: path too short to extract id_folder and insole_file_name: {folder_path}")
        return

    # Extract identifiers
    id_folder = parts[-2]
    insole_file_name = parts[-1]    

    # Prepare output directories
    resize_dir = os.path.join(output_path, id_folder, insole_file_name, f'Resize_{start_frame}_{end_frame}')
    step_dir   = os.path.join(output_path, id_folder, insole_file_name, f'Step_{start_frame}_{end_frame}')
    extract_dir   = os.path.join(output_path, id_folder, insole_file_name, f'Extract_{start_frame}_{end_frame}')
    os.makedirs(resize_dir, exist_ok=True)
    os.makedirs(step_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    # Pattern to extract frame number
    frame_pattern = re.compile(r"_(\d+)\.png$", re.IGNORECASE)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.png'):
            continue
        m = frame_pattern.search(filename)
        if not m:
            continue
        frame = int(m.group(1))
        if frame < start_frame or frame > end_frame:
            continue

        # Load grayscale image
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('L')
        arr = np.array(img, dtype=np.uint8) * 1.2

        # Resize 10x
        h, w = arr.shape
        new_size = (w * 5, h * 5)
        resized = img.resize(new_size, Image.BILINEAR)
        arr_resized = np.array(resized, dtype=np.uint8)

        # Apply false-color map (e.g., 'jet'), but set zero values to black
        normalized = arr_resized.astype(np.float32) / 255.0
        cmap = cm.get_cmap('jet')
        colored = cmap(normalized)  # RGBA floats [0,1]
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        # Map zeros to black
        mask_zero = (arr_resized == 0)
        colored_rgb[mask_zero] = [0, 0, 0]
        colored_pil = Image.fromarray(colored_rgb)

        # Save colored image
        colored_pil.save(os.path.join(resize_dir, filename))

        # Quantize into 8 steps
        quantized = (arr_resized // 32) * 32  # levels: 0,32,...,224
        norm_quant = quantized.astype(np.float32) / 255.0
        colored_step = cmap(norm_quant)
        colored_step_rgb = (colored_step[:, :, :3] * 255).astype(np.uint8)
        # Ensure zeros remain black
        colored_step_rgb[quantized == 0] = [0, 0, 0]
        Image.fromarray(colored_step_rgb).save(os.path.join(step_dir, filename))
        extract_frame = sample_frames(start_sample, end_sample)
        if frame in extract_frame:
            Image.fromarray(colored_step_rgb).save(os.path.join(extract_dir, filename))

    # --- New: build video from resized frames ---
    # Collect and sort frame filenames
    frames = []
    for fn in os.listdir(step_dir):
        m = frame_pattern.search(fn)
        if m:
            frames.append((int(m.group(1)), fn))
    frames.sort(key=lambda x: x[0])

    if frames:
        # Determine video properties
        first_frame = cv2.imread(os.path.join(step_dir, frames[0][1]))
        height, width, _ = first_frame.shape
        fps = 1 / 0.04  # 50 fps
        parent_path = os.path.dirname(step_dir)
        video_path = os.path.join(parent_path, f"{id_folder}_{start_frame}_{end_frame}.mp4")

        # Define the codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Write each frame
        for _, fn in frames:
            img = cv2.imread(os.path.join(step_dir, fn))
            out.write(img)

        out.release()
        print(f"Video saved to: {video_path}")
    else:
        print("No resized frames found; video not created.")
    """ 
    print(f"Processed frames {start_frame} to {end_frame}.\n"
          f"Resized+colored images in: {resize_dir}\n"
          f"Quantized images in: {step_dir}") """
    
def sample_frames(a, b, n=14):
    """
    Return a list of `n` integers evenly spaced from a to b (inclusive).
    The first element is exactly a, the last is exactly b.
    """
    if n < 2:
        raise ValueError("n must be at least 2 to include both endpoints")
    # Compute the step in float, then round each sample to nearest int
    step = (b - a) / (n - 1)
    frames = [int(round(a + i * step)) for i in range(n)]
    
    # Fix any rounding drift to ensure exact endpoints
    frames[0] = a
    frames[-1] = b
    return frames
