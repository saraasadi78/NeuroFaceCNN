import os
import numpy as np

# Set the label: "face" or "noface"
label = "face"

# Root directory containing subject folders
root_dir = fr"D:\sarafiles\Face-Project\face-noface-original-txt\{label}"

# Where to save the converted .npy results
save_dir = fr"D:\sarafiles\Face-Project\data-npy-seperately-for-deepexplain-one-by-one\{label}"
os.makedirs(save_dir, exist_ok=True)

# Loop through each subject folder
for sub_folder in os.listdir(root_dir):
    sub_path = os.path.join(root_dir, sub_folder)
    if not os.path.isdir(sub_path):
        continue

    print(f"Processing {label} - {sub_folder}...")

    # Create subject-specific save folder
    sub_save_dir = os.path.join(save_dir, sub_folder)
    os.makedirs(sub_save_dir, exist_ok=True)

    # Loop through .txt files in this subject folder
    for file in os.listdir(sub_path):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(sub_path, file)
        try:
            # Load comma-separated numeric data
            arr = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
            if arr.shape != (41489, 10):
                print(f" Skipping {file_path}: unexpected shape {arr.shape}")
                continue

            # Save as .npy inside subject folder
            save_name = f"{os.path.splitext(file)[0]}.npy"
            save_path = os.path.join(sub_save_dir, save_name)
            np.save(save_path, arr)
            print(f" Saved {label} file: {save_path}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

print(f" Conversion complete — files saved per subject for label: {label}")
