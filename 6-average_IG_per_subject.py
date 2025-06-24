# Computes per-second average of IG-reverted NIfTI samples for each subject and saves as a single 4D file

import os
import nibabel as nib
import numpy as np

label = "face"

# Define input and output directories
pickle_path = rf"E:\sara.asadi\IG-pre‐sigmoid-logit-backed-to-brain\{label}"
save_dir = rf"E:\sara.asadi\IG-pre‐sigmoid-logit-backed-to-brain-avged\{label}"

# List of subject directories to process
subjects = [f"IG-reverted_{label}_sub{i}" for i in range(1, 87)]

# Loop over all subjects
for sub in subjects:
    sub_dir = os.path.join(pickle_path, sub)
    nii_files = [os.path.join(sub_dir, file) for file in os.listdir(sub_dir) if file.endswith('.nii.gz')]

    if not nii_files:
        print(f"No NIfTI files found for {sub}. Skipping.")
        continue

    # Load a sample image to get shape and affine info
    img = nib.load(nii_files[0])
    averages_per_second = []

    # Loop over time dimension (assumed 10 time points)
    for t in range(10):
        total_sum = None
        for file in nii_files:
            img = nib.load(file)
            img_data = img.get_fdata()
            img_data_time = img_data[..., t]
            if total_sum is None:
                total_sum = np.zeros_like(img_data_time)
            total_sum += img_data_time

        # Average across all samples for time t
        average_data = total_sum / len(nii_files)
        averages_per_second.append(average_data)

    # Convert list to 4D array: shape (x, y, z, t)
    averages_per_second = np.array(averages_per_second)
    new_arr = np.transpose(averages_per_second, (1, 2, 3, 0))

    # Save the averaged result as a new NIfTI file
    average_img = nib.Nifti1Image(new_arr, affine=img.affine, header=img.header)
    output_nifti_path = os.path.join(save_dir, f"{sub}-{label}-averaged.nii.gz")
    nib.save(average_img, output_nifti_path)

    print(f"Saved average image for subject {sub} ({label}) → {output_nifti_path}")
