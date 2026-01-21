import os
import csv
from PIL import Image
import numpy as np
import re

folder_path = input("Enter prediction directory path: "); print("Valid path." if os.path.exists(folder_path) else "Invalid path.")
# Determine project root (one level above this script's directory)
output_path = os.path.dirname(os.path.abspath(__file__))
start_frame = 0
end_frame = 250

def write_csv(csv_file, id_folder, insole_file_name, side, overall_max, contact_area):
    file_exists = os.path.isfile(csv_file)
    # Append result
    with open(csv_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['id_folder', 'insole_file_name', 'side', 'max_value', 'contact area'])
        writer.writerow([id_folder, insole_file_name, side, overall_max, contact_area])
    print(f"Appended values for {insole_file_name} to {csv_file}")

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

    # Scan grayscale images
    overall_max = -1
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
            arr = np.array(img)
            curr_max = arr.max()
            if curr_max > overall_max:
                overall_max = curr_max
            found = True
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if not found:
        print(f"No grayscale PNG images found in {folder_path}")
        return

    # Prepare CSV path in parent of id_folder
    csv_dir = os.path.join(output_path, id_folder, insole_file_name)
    csv_file = os.path.join(csv_dir, id_folder + '_report.csv')
    os.makedirs(csv_dir, exist_ok=True)
    #print(f"Processed folder {folder_path}: max pixel value = {overall_max}")
    return csv_file, id_folder, insole_file_name, side, overall_max
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
    img_file = os.path.join(img_dir, id_folder + '_contact.png')
    os.makedirs(img_dir, exist_ok=True)
    
    # Save the binary image
    Image.fromarray(binary_image).save(img_file)
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

            if sum_matrix is None:
                sum_matrix = arr
            else:
                sum_matrix += arr
            image_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if image_count == 0:
        print(f"No valid PNG frames in [{start_frame}–{end_frame}] found in {folder_path}")
        return

    # Clip and scale
    clipped = np.clip(sum_matrix, 0, np.max(sum_matrix))
    scaled = (clipped * (255.0 / np.max(sum_matrix))).astype(np.uint8)
    scaled = np.where(scaled < 1.5, 0, scaled)

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
    np.savetxt(out_csv, sum_matrix / 25, delimiter=",", fmt="%d")

    out_gray = os.path.join(img_dir, f"{id_folder}_{side}_time_integral_{start_frame}-{end_frame}_g.png")
    out_jet  = os.path.join(img_dir, f"{id_folder}_{side}_time_integral_{start_frame}-{end_frame}_h.png")
    image_gray.save(out_gray)
    image_jet.save(out_jet)

    print(f"Processed {image_count} frames in [{start_frame}–{end_frame}].")
    print(f"Saved CSV to {out_csv}")
    print(f"Saved greyscale map to {out_gray}")
    print(f"Saved heatmap to {out_jet}")


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
    Image.fromarray(new_img).save(out_img)

    # save CSV of (x, y)
    csv_path = os.path.join(img_dir, f"{id_folder}_{side}_COP_{start_frame}-{end_frame}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])
        for r, c in centroids:
            writer.writerow([c, r])

    print(f"Saved CoP image to {out_img}")
    print(f"Saved CoP coords to {csv_path}")

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
            pixel_sum = int(arr.sum())
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
    forces_N  = [s / 25 for s in sums]     # calibration: sum/25 → N

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

    print(f"Saved GRF plot to {plot_path}")
    print(f"Saved GRF data to {csv_path}")

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
    print(f"Saved 3D heatmap scatter for frames {start_frame}–{end_frame} to {out_path}")

csv_file, id_folder, insole_file_name, side, overall_max = max_pixel(folder_path, output_path)
csv_file, id_folder, insole_file_name, side, contact_area = process_contact_area(folder_path, output_path)
time_integral_image(folder_path, output_path, start_frame, end_frame)
write_csv(csv_file, id_folder, insole_file_name, side, overall_max, contact_area)
centre_pressure_image(folder_path, output_path, start_frame, end_frame)
ground_reaction_force(folder_path, output_path, start_frame, end_frame)
plot_3d_heatmap(folder_path, output_path, start_frame, end_frame)