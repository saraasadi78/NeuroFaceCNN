import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

#   CONFIG  
label = "face"  # or "noface"
base_dir = r"D:\sarafiles\Face-Project"

mask_path = os.path.join(base_dir, "85_subBrainMask_average_99.nii.gz")
input_dir = os.path.join(base_dir, f"D:\sarafiles\Face-Project\Ig\{label}")
output_dir = os.path.join(base_dir, f"IG_npy_back_to_brain\\{label}")
os.makedirs(output_dir, exist_ok=True)

SHOW_QC = False  # visualization toggle

#   LOAD MASK  
if not os.path.exists(mask_path):
    raise FileNotFoundError(f"Mask not found: {mask_path}")

mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()
mask_shape = mask_data.shape
mask_flat = mask_data.reshape(-1)
mask_indices = np.nonzero(mask_flat)[0]
print(f"Loaded mask: shape={mask_shape}, active voxels={mask_indices.size:,}")


#   FUNCTION  
def revert_to_brain(voxel_matrix: np.ndarray,
                    mask_shape: tuple,
                    mask_indices: np.ndarray,
                    reference_affine: np.ndarray) -> nib.Nifti1Image:
   
    #Reconstructs voxel×time matrix into full 3D+time NIfTI
    n_vox, n_time = voxel_matrix.shape
    if n_vox != len(mask_indices):
        raise ValueError(f"Voxel mismatch: data={n_vox}, mask={len(mask_indices)}")

    brain_4d = np.zeros((*mask_shape, n_time), dtype=np.float32)
    for t in range(n_time):
        flat_vol = np.zeros(mask_flat.shape, dtype=np.float32)
        flat_vol[mask_indices] = voxel_matrix[:, t]
        brain_4d[..., t] = flat_vol.reshape(mask_shape)

    return nib.Nifti1Image(brain_4d, affine=reference_affine)

#   MAIN LOOP  
sub_folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
if not sub_folders:
    print(f"No subject folders found in {input_dir}")
else:
    print(f"Found {len(sub_folders)} subject folders.\n")

for sub_path in sub_folders:
    subj_name = os.path.basename(sub_path)
    npy_files = [f for f in os.listdir(sub_path) if f.endswith(".npy")]
    if not npy_files:
        print(f"No .npy files in {subj_name}, skipping.")
        continue

    for fname in npy_files:
        fpath = os.path.join(sub_path, fname)
        print(f"Loading {subj_name}/{fname} ...")
        data = np.load(fpath)
        data = np.squeeze(data)

        if data.ndim != 2 or data.shape[0] != mask_indices.size:
            print(f"Shape {data.shape} doesn’t match expected (41489,10). Skipping {fname}.")
            continue

        # Convert to NIfTI
        nifti_img = revert_to_brain(data, mask_shape, mask_indices, mask_img.affine)

        # Save output in the same subject’s folder
        subj_out_dir = os.path.join(output_dir, subj_name)
        os.makedirs(subj_out_dir, exist_ok=True)
        out_path = os.path.join(subj_out_dir, f"{os.path.splitext(fname)[0]}_backToBrain.nii.gz")
        nib.save(nifti_img, out_path)
        print(f"Saved: {out_path} | shape={nifti_img.shape}")

        if SHOW_QC:
            mean_vol = np.mean(nifti_img.get_fdata(), axis=-1)
            mid = mask_shape[2] // 2
            plt.imshow(mean_vol[:, :, mid], cmap='hot', origin='lower')
            plt.title(f"{subj_name} | mean across 10 TRs | slice {mid}")
            plt.axis('off')
            plt.show()

print("\nReconstruction completed for all subjects.")
