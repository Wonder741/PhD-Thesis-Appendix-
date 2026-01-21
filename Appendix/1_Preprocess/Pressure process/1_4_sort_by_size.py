import os
import csv
import shutil
from pathlib import Path


def build_id_to_leftcm(csv_sample: str) -> dict:
    """
    Build lookup: id1 (alphabetic ID) -> left_cm (float)
    """
    id_to_leftcm = {}
    with open(csv_sample, newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # skip header
        for row in reader:
            if len(row) < 7:
                continue
            id1 = row[1].strip()
            try:
                left_cm = float(row[5])
            except ValueError:
                continue
            id_to_leftcm[id1] = left_cm
    return id_to_leftcm


def map_left_cm_to_folder(left_cm: float) -> str:
    """
    Discretize left_cm into one of three categories:
      <240  -> "240"
      240–260 -> "260"
      >260  -> "280"
    """
    if left_cm < 240:
        return "240"
    elif left_cm > 260:
        return "280"
    else:
        return "260"


def sort_greyscale_by_leftcm(
    csv_sample: str,
    source_root: str,
    target_root: str,
):
    """
    Source:
      D:\\Appendix\\Pressure\\E_Registor_Greyscale\\{num}{id}\\{side}\\*.png

    Target:
      D:\\Appendix\\Pressure\\F_Sorted_Greyscale\\{240|260|280}\\{num}{id}\\{side}\\*.png
    """
    source_root = Path(source_root)
    target_root = Path(target_root)

    id_to_leftcm = build_id_to_leftcm(csv_sample)

    for num_id_dir in source_root.iterdir():
        if not num_id_dir.is_dir():
            continue

        # Folder name: {num}{id}
        num_id = num_id_dir.name
        id_alpha = "".join(filter(str.isalpha, num_id))

        left_cm = id_to_leftcm.get(id_alpha)
        if left_cm is None:
            print(f"Skip folder {num_id}: ID '{id_alpha}' not found in CSV")
            continue

        # 🔑 discretized folder
        left_cm_folder = map_left_cm_to_folder(left_cm)

        for side_dir in num_id_dir.iterdir():
            if not side_dir.is_dir():
                continue

            side = side_dir.name

            for img_path in side_dir.glob("*.png"):
                dst_path = (
                    target_root
                    / left_cm_folder
                    / num_id
                    / side
                    / img_path.name
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dst_path)

    print("Sorting complete!")


if __name__ == "__main__":
    CSV_SAMPLE = r"...\sample_data.csv"

    SOURCE_ROOT = r"...\E_Registor_Greyscale"
    TARGET_ROOT = r"...\F_Sorted_Greyscale"

    sort_greyscale_by_leftcm(
        csv_sample=CSV_SAMPLE,
        source_root=SOURCE_ROOT,
        target_root=TARGET_ROOT,
    )
