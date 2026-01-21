from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

factor = 2
def average_batches_to_greyscale_images(
    input_root: str,
    output_root: str,
    batch_size: int = 20,
):
    """
    Merge of:
      - Sequence 5: batch-average matrices (left/right separately)
      - Sequence 6: convert matrices to grayscale PNG

    Pipeline (per folder, per side):
      1) Find CSVs named: {folder}_{num1}_{num2}_{side}.csv
      2) Group by num1, sort by num2
      3) Average every `batch_size` matrices element-wise
      4) Clip to [0,255], convert to uint8
      5) Save as grayscale PNG:
           {folder}_{num1}_{side}_{batch_idx}.png

    Notes:
      - No intermediate averaged CSV is saved.
      - Assumes each CSV is a 64x64 matrix without headers.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    for folder in input_root.iterdir():
        if not folder.is_dir():
            continue

        for side in ["left", "right"]:
            in_dir = folder / side
            if not in_dir.exists():
                continue

            out_dir = output_root / folder.name / side
            out_dir.mkdir(parents=True, exist_ok=True)

            # Example expected stem:
            #   {folder}_{num1}_{num2}_left
            # or {folder}_{num1}_{num2}_right
            files = sorted(in_dir.glob(f"{folder.name}_*_*_{side}.csv"))

            # Group by num1
            groups = {}
            for f in files:
                parts = f.stem.split("_")
                # expected: [folder, num1, num2, side]
                if len(parts) != 4:
                    continue

                _, num1, num2, last = parts
                if last != side:
                    continue

                try:
                    idx2 = int(num2)  # frame index
                except ValueError:
                    continue

                groups.setdefault(num1, []).append((idx2, f))

            # Process each group
            for num1, entries in groups.items():
                # sort by frame index (num2)
                entries.sort(key=lambda x: x[0])

                # batch average
                for batch_idx, i in enumerate(range(0, len(entries), batch_size), start=1):
                    batch = entries[i:i + batch_size]
                    if len(batch) < batch_size:
                        break  # match your original behavior: ignore incomplete last batch

                    mats = []
                    for _, fp in batch:
                        mat = pd.read_csv(fp, header=None).values.astype(float)
                        mats.append(mat)

                    avg_mat = np.mean(np.stack(mats, axis=2), axis=2)

                    # --- FIX: handle NaN / Inf before casting ---
                    avg_mat = np.nan_to_num(
                        avg_mat,
                        nan=0.0,
                        posinf=255.0,
                        neginf=0.0
                    )

                    # ---- Apply scaling factor = 2 ----
                    # 512 (or larger) → 255
                    scaled = avg_mat / factor

                    img_arr = np.clip(scaled, 0, 255).astype(np.uint8)

                    # save grayscale image
                    img = Image.fromarray(img_arr, mode="L")
                    out_name = f"{folder.name}_{num1}_{side}_{batch_idx}.png"
                    img.save(out_dir / out_name)

        print(f"Finished: {folder.name}")


if __name__ == "__main__":
    ROOT_INPUT = r"...\B_Split_Data"       # contains {folder}/left and {folder}/right CSVs
    ROOT_OUTPUT = r"...\C_Split_Greyscale"  # final output: PNG only

    average_batches_to_greyscale_images(
        input_root=ROOT_INPUT,
        output_root=ROOT_OUTPUT,
        batch_size=20,
    )
