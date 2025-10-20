# Computes per-second average of IG-reverted NIfTI samples for each subject and saves as a single 4D file
import os
import nibabel as nib
import numpy as np

label = "face"

base_dir = rf"D:\sarafiles\Face-Project\IG_npy_back_to_brain\{label}"
save_dir = rf"D:\sarafiles\Face-Project\Ig-avged\{label}"
os.makedirs(save_dir, exist_ok=True)

# Helper: list NIfTI files (both .nii and .nii.gz)
def list_niftis(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".nii") or f.lower().endswith(".nii.gz")
    ]

# Find subject folders
subjects = [
    d for d in os.listdir(base_dir)
    if d.startswith("sub-") and os.path.isdir(os.path.join(base_dir, d))
]
subjects.sort(key=lambda s: int(s.split("-")[1]))  # numeric sort by sub id if possible

for sub in subjects:
    sub_dir = os.path.join(base_dir, sub)
    nii_files = list_niftis(sub_dir)
    if not nii_files:
        print(f"[WARN] No NIfTI files in {sub_dir}; skipping.")
        continue

    # Probe first file for shape/affine/T
    img0 = nib.load(nii_files[0])
    data0 = img0.get_fdata()
    if data0.ndim != 4:
        print(f"[WARN] {nii_files[0]} is not 4D; skipping {sub}.")
        continue

    X, Y, Z, T = data0.shape
    if T != 10:
        print(f"[INFO] {sub}: detected T={T} (not assuming 10).")

    # Sanity: all files same shape
    bad = []
    for f in nii_files[1:]:
        shp = nib.load(f).shape
        if shp != (X, Y, Z, T):
            bad.append((f, shp))
    if bad:
        print(f"[ERROR] Shape mismatch in {sub}. First: {(X,Y,Z,T)}; mismatches: {bad[:3]}...")
        print("        Fix inputs and rerun. Skipping this subject.")
        continue

    # Accumulate sum per timepoint in a single pass
    sum_TXYZ = np.zeros((T, X, Y, Z), dtype=np.float64)
    for f in nii_files:
        d = nib.load(f).get_fdata().astype(np.float64, copy=False)
        # guard NaNs/Infs
        np.nan_to_num(d, copy=False)
        # move time axis first -> (T, X, Y, Z) and add
        sum_TXYZ += np.moveaxis(d, -1, 0)

    mean_TXYZ = sum_TXYZ / len(nii_files)
    mean_XYZT = np.moveaxis(mean_TXYZ, 0, -1).astype(np.float32)

    # Preserve affine/header from first image
    out_img = nib.Nifti1Image(mean_XYZT, affine=img0.affine, header=img0.header)
    out_name = f"{sub}_{label}_averaged.nii.gz"
    out_path = os.path.join(save_dir, out_name)
    nib.save(out_img, out_path)

    print(f"[OK] Saved {sub} ({len(nii_files)} files, T={T}) → {out_path}")
