# Visualizes averaged Integrated Gradients maps on cortical surfaces for face vs. no-face conditions

import os
import shutil
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from nilearn import plotting, surface, datasets, image
from matplotlib.colors import Normalize

# Paths to averaged 4D NIfTI files for one subject
data_paths = {
    "Face": "sara.asadi/IG-reverted_face_subxx-averaged.nii.gz",
    "No Face": "sara.asadi/IG-reverted_noface_subxx-averaged.nii.gz"}

# Load FreeSurfer fsaverage surface (inflated and pial)
fs = datasets.fetch_surf_fsaverage()
temp_dir = "temp_surf_images"
os.makedirs(temp_dir, exist_ok=True)

# Load NIfTI files
face_nii = nib.load(data_paths["Face"])
noface_nii = nib.load(data_paths["No Face"])
n_tp = 10  # Number of timepoints

# Create subplot canvas
fig, axes = plt.subplots(n_tp, 2, figsize=(6, 2 * n_tp), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

# Loop over timepoints
for t in range(n_tp):
    face_data = image.index_img(face_nii, t).get_fdata()
    noface_data = image.index_img(noface_nii, t).get_fdata()

    face_pos = np.clip(face_data, 0, None)
    face_neg = np.abs(np.clip(face_data, None, 0))
    noface_pos = np.clip(noface_data, 0, None)
    noface_neg = np.abs(np.clip(noface_data, None, 0))

    # Plot both conditions: face and no-face
    for col, (title, combined_data, cmap) in enumerate([
        ("Face Input", face_pos + noface_neg, "Reds"),
        ("No Face Input", noface_pos + face_neg, "Blues")
    ]):
        combined_data = combined_data.astype(np.float32)
        cmin, cmax = combined_data.min(), combined_data.max()
        if cmax > cmin:
            combined_data = (combined_data - cmin) / (cmax - cmin)
        else:
            combined_data = np.zeros_like(combined_data)

        crops = []
        for hemi in ["left", "right"]:
            arr = nib.Nifti1Image(combined_data, face_nii.affine)
            tex = surface.vol_to_surf(arr, fs.pial_left if hemi == "left" else fs.pial_right)

            fn = os.path.join(temp_dir, f"{title.replace(' ', '_')}_{hemi}_t{t}.png")
            thresh = np.percentile(np.abs(tex[tex != 0]), 84) if np.any(tex != 0) else 0

            plotting.plot_surf_stat_map(
                fs.infl_left if hemi == "left" else fs.infl_right,
                tex,
                hemi=hemi,
                view="ventral",
                threshold=thresh,
                cmap=cmap,
                bg_map=fs.sulc_left if hemi == "left" else fs.sulc_right,
                colorbar=False,
                figure=plt.figure(dpi=150)
            )
            plt.savefig(fn, bbox_inches="tight")
            plt.close()

            img = plt.imread(fn)[:, :, :3]
            img = np.rot90(img, 3)
            img = np.fliplr(img)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = (gray < 0.98).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
            crop = img[y:y + h, x:x + w] if w > 0 and h > 0 else np.zeros((10, 10, 3))
            crops.append(crop)

        h_min = min(c.shape[0] for c in crops)
        w_min = min(c.shape[1] for c in crops)
        combined_img = np.concatenate([
            cv2.resize(crops[0], (w_min, h_min)),
            cv2.resize(crops[1], (w_min, h_min))
        ], axis=1)

        ax = axes[t, col]
        ax.imshow(combined_img)
        ax.axis("off")
        if col == 0:
            ax.text(-0.1, 0.5, f"T={t + 1}", va="center", ha="right", fontsize=10, fontweight="bold", transform=ax.transAxes)
        if t == 0:
            ax.set_title(title, fontsize=12, fontweight="bold")

# Add colorbars
cax1 = fig.add_axes([0.92, 0.55, 0.02, 0.3])
plt.colorbar(plt.cm.ScalarMappable(cmap="Reds"), cax=cax1).set_label("Face Attribution", fontsize=8, fontweight="bold")

cax2 = fig.add_axes([0.92, 0.15, 0.02, 0.3])
plt.colorbar(plt.cm.ScalarMappable(cmap="Blues"), cax=cax2).set_label("No-Face Attribution", fontsize=8, fontweight="bold")

# Save final figure
fig.savefig("sara.asadi/IG-subxx-ventral.png", dpi=500, bbox_inches="tight")
plt.show()

# Remove temporary surface image directory
shutil.rmtree(temp_dir)
