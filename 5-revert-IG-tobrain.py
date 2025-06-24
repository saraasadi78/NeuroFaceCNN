"""
Reconstructs voxel-level Integrated Gradients (IG) maps into 4D NIfTI images.
This script:
- Loads the group-level binary brain mask.
- Loads subject-specific IG attributions stored as pickled numpy arrays.
- Reconstructs each sample (shape: [41489, 10]) back to full 3D brain (85x85x85) over time.
- Saves each reconstructed sample as a 4D NIfTI file.
"""

import os
import numpy as np
import nibabel as nib
import pickle


label = "face"
base_path = r"E:\sara.asadi\pypro"
brain_mask_file = os.path.join(base_path, "85_subBrainMask_average_99.nii.gz")
ig_input_path   = rf"E:\sara.asadi\IG-pre‐sigmoid-logit\{label}"
output_dir      = rf"E:\sara.asadi\IG-pre‐sigmoid-logit-backed-to-brain\{label}"

#Load Brain Mask
if not os.path.exists(brain_mask_file):
    raise FileNotFoundError(f"Brain mask file not found: {brain_mask_file}")

brain_mask = nib.load(brain_mask_file)
mask_data = brain_mask.get_fdata()
mask_shape = mask_data.shape
print(f"Brain mask loaded with shape: {mask_shape}")

mask_flat = mask_data.reshape(-1)
nonzero_indices = np.nonzero(mask_flat)[0]
print(f"Nonzero voxels in brain mask: {nonzero_indices.shape[0]}")


#Function to reconstruct full brain volume from masked IG data
def revert_to_brain(ig_matrix, mask_shape, mask_indices):

    if ig_matrix.shape[0] != mask_indices.shape[0]:
        raise ValueError("Mismatch between IG matrix and brain mask indices.")

    brain_4d = np.zeros((*mask_shape, ig_matrix.shape[1]))  # e.g., (85, 85, 85, 10)

    for t in range(ig_matrix.shape[1]):
        flat_volume = np.zeros(mask_flat.shape)
        flat_volume[mask_indices] = ig_matrix[:, t]
        brain_4d[..., t] = flat_volume.reshape(mask_shape)

    return nib.Nifti1Image(brain_4d, affine=brain_mask.affine)

# Loop through subjects
for subj in range(1, 87):
    pickle_file = os.path.join(ig_input_path, f"IG_sub-{subj}_{label}.pickle")
    
    if not os.path.exists(pickle_file):
        print(f"File missing for subject {subj}. Skipping.")
        continue

    print(f"\n Loading IG data for subject {subj}...")
    with open(pickle_file, 'rb') as f:
        ig_data = pickle.load(f)  # Shape: (N_samples, 41489, 10, 1)

    ig_data = ig_data.squeeze()  # Shape: (N_samples, 41489, 10)
    print(f"   Loaded shape: {ig_data.shape}")

    if ig_data.shape[1] != nonzero_indices.shape[0]:
        print(f"Warning: Voxel mismatch (data={ig_data.shape[1]}, mask={nonzero_indices.shape[0]})")
        continue

    # Create subject output directory
    subj_output_dir = os.path.join(output_dir, f"IG-reverted_{label}_sub{subj}")
    os.makedirs(subj_output_dir, exist_ok=True)

    for sample_idx in range(ig_data.shape[0]):
        sample = ig_data[sample_idx]  # Shape: (41489, 10)
        print(f"Processing sample {sample_idx} of subject {subj}...")

        img_nifti = revert_to_brain(sample, mask_shape, nonzero_indices)

        filename = f"IG-reverted_sub_{subj}_{sample_idx}_{label}.nii.gz"
        output_path = os.path.join(subj_output_dir, filename)
        nib.save(img_nifti, output_path)

        print(f"Saved: {output_path} | Shape: {img_nifti.shape}")

    print(f"Completed subject {subj}.\n")

