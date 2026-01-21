from pathlib import Path
import pandas as pd


def process_raw_csv_to_left_right(input_root: str, output_root: str):
    """
    End-to-end preprocessing pipeline for raw plantar pressure CSV files.

    For each raw CSV file in the input directory:
    1. Skip the global file header (23 rows)
    2. Split remaining data into frames (98 rows per frame)
    3. For each frame:
       - Skip frame-level metadata (2 rows)
       - Parse numeric matrix
       - Remove the first column (non-sensor data)
       - Split into upper and lower halves (48 rows each)
       - Pad each half with zero rows to reach 64×64
       - Save final left/right matrices only

    Parameters
    ----------
    input_root : str
        Root directory containing subfolders with raw CSV files.
    output_root : str
        Root directory where processed left/right CSV files are saved.

    Output Structure
    ----------------
    output_root/
      └── subject_x/
          ├── left/
          │   └── *_left.csv
          └── right/
              └── *_right.csv
    """

    # Convert input paths to Path objects
    input_root = Path(input_root)
    output_root = Path(output_root)

    # Pre-create an 8×64 zero matrix for padding
    # This will be added to the top and bottom of each half-frame
    zero_block = pd.DataFrame([[0] * 64 for _ in range(8)])

    # Iterate through each subject/session folder
    for folder in input_root.iterdir():
        if not folder.is_dir():
            continue

        # Create output folders for left and right matrices
        left_dir = output_root / folder.name / "left"
        right_dir = output_root / folder.name / "right"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)

        # Process each raw CSV file in the folder
        for csv_file in sorted(folder.glob("*.csv")):

            # Read file as raw text to preserve frame structure
            with csv_file.open("r") as f:
                lines = f.readlines()

            # --------------------------------------------------
            # Step 1: Remove global file header
            # --------------------------------------------------
            data_lines = lines[23:]

            # Determine number of complete frames in the file
            n_frames = len(data_lines) // 98

            # --------------------------------------------------
            # Step 2: Process each frame independently
            # --------------------------------------------------
            for frame_idx in range(n_frames):

                # Extract one frame (98 rows)
                frame_lines = data_lines[
                    frame_idx * 98 : (frame_idx + 1) * 98
                ]

                # --------------------------------------------------
                # Step 3: Remove frame-level metadata
                # --------------------------------------------------
                # First 2 rows are not sensor data
                sensor_lines = frame_lines[2:]

                # Parse numeric values from text
                matrix = [
                    list(map(int, line.strip().split()))
                    for line in sensor_lines
                ]

                df = pd.DataFrame(matrix)

                # --------------------------------------------------
                # Step 4: Remove non-sensor column
                # --------------------------------------------------
                # Drop the first column (index / timestamp)
                df = df.iloc[:, 1:]

                # Safety check: enforce expected matrix size
                if df.shape != (96, 64):
                    raise ValueError(
                        f"{csv_file.name}, frame {frame_idx + 1}: "
                        f"unexpected shape {df.shape}"
                    )

                # --------------------------------------------------
                # Step 5: Split into upper and lower regions
                # --------------------------------------------------
                # Ensure exactly 64 columns (drop extras if any)
                df = df.iloc[:, :64].copy()

                # Normalize column labels to 0..63 so concat can't create an extra column
                df.columns = range(64)
                upper_half = df.iloc[:48, :]
                lower_half = df.iloc[48:, :]

                # --------------------------------------------------
                # Step 6: Zero-padding to 64×64
                # --------------------------------------------------
                upper_padded = pd.concat(
                    [zero_block, upper_half, zero_block],
                    ignore_index=True
                )

                lower_padded = pd.concat(
                    [zero_block, lower_half, zero_block],
                    ignore_index=True
                )

                # --------------------------------------------------
                # Step 7: Save final outputs
                # --------------------------------------------------
                base_name = f"{csv_file.stem}_{frame_idx + 1}"

                upper_padded.to_csv(
                    left_dir / f"{base_name}_left.csv",
                    index=False,
                    header=False
                )

                lower_padded.to_csv(
                    right_dir / f"{base_name}_right.csv",
                    index=False,
                    header=False
                )

        print(f"Finished processing folder: {folder.name}")


if __name__ == "__main__":
    # Root directory containing raw CSV data
    ROOT_INPUT = r"...\A_Raw_Data"

    # Root directory for final processed output
    ROOT_OUTPUT = r"...\B_Split_Data"

    process_raw_csv_to_left_right(ROOT_INPUT, ROOT_OUTPUT)
