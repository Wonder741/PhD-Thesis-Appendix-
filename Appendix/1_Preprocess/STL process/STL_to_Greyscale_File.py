#!/usr/bin/env python3
import A_STL_to_Greyscale_File
import B_3dscan_image_registor_File
import os
import shutil
from PIL import Image
import sys

# Input: get directory path from user
input_dir = input("Enter directory path containing STL files: ").strip()
if not os.path.isdir(input_dir):
    print(f"Error: {input_dir} is not a valid directory.")
    sys.exit(1)

# Determine input directory's lowest-level folder name
lowest_folder = os.path.basename(os.path.normpath(input_dir))

# Determine project root (one level above this script's directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Base STL staging folder including lowest-level folder
base_stl_folder = os.path.join(script_dir, "1STL", lowest_folder)
if not os.path.exists(base_stl_folder):
    os.makedirs(base_stl_folder, exist_ok=True)

# Copy all .stl files from input_dir to base_stl_folder
for fname in os.listdir(input_dir):
    if fname.lower().endswith(".stl"):
        shutil.copy2(
            os.path.join(input_dir, fname),
            os.path.join(base_stl_folder, fname)
        )
        print(f"Copied {fname} to {base_stl_folder}")

# Processing parameters
output_resolution = 320

# Process all STL files in base_stl_folder only
for fname in os.listdir(base_stl_folder):
    if not fname.lower().endswith(".stl"):
        continue
    name_no_ext = fname[:-4]
    if name_no_ext.endswith("_L"):
        side = "left"
        id_folder = lowest_folder
    elif name_no_ext.endswith("_R"):
        side = "right"
        id_folder = lowest_folder
    else:
        # ignore files without _L or _R suffix
        continue

    # Create target STL folder structure
    target_stl_folder = os.path.join(base_stl_folder)
    dst_file = os.path.join(target_stl_folder, fname)

    # Prepare output directories under project root
    greyscale_folder = os.path.join(script_dir, "2Greyscale", id_folder, side)
    debug_base_folder = os.path.join(script_dir, "3Debug", id_folder, side)
    processed_folder = os.path.join(script_dir, "4Processed", id_folder, side)
    resized_folder = os.path.join(script_dir, "5Resized", id_folder, side)
    for folder in [greyscale_folder, debug_base_folder, processed_folder, resized_folder]:
        os.makedirs(folder, exist_ok=True)

    # Convert STL to greyscale PNG
    png_file_name = f"{side}_{id_folder}.png"
    greyscale_file = os.path.join(greyscale_folder, png_file_name)
    footDimension = A_STL_to_Greyscale_File.process_stl_to_png(
        dst_file, greyscale_file, output_resolution
    )
    print(f"Foot dimensions for {fname}: {footDimension}")

    # Align and process image
    size_str = str(int(footDimension[1] / 10 + 1))
    insole_name = f"{side}_{size_str}.png"
    object_image_file = os.path.join(script_dir, "0Insoles", side, insole_name)
    processed_points_img = B_3dscan_image_registor_File.align_loop(
        object_image_file, greyscale_file, debug_base_folder, processed_folder
    )

    # Resize processed image and save
    processed_img = Image.fromarray(processed_points_img)
    processed_img = processed_img.resize((64, 64), Image.LANCZOS)
    resized_file_path = os.path.join(resized_folder, png_file_name)
    processed_img.save(resized_file_path)
    print(f"Finished processing {fname}")
