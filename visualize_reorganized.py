import os
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, surface, datasets, image
from scipy.ndimage import binary_erosion

# Config
data_paths = {
    "Face": "/Users/saraasadi/Desktop/face-noface/group_face_averaged.nii.gz",
    "NoFace": "/Users/saraasadi/Desktop/face-noface/group_noface_averaged.nii.gz",
}

ffa_mask_path = "/Users/saraasadi/Desktop/qqqqqq.nii.gz"
ffa_threshold = 0.5  # keep hard-labeled ROI vertices
erosion_iter = 2  # shrink ROI a bit
display_percentile = 95  # scaling for surf maps
out_png = "group_averaged-lastIG-95treshold-ventral.png"
cmap = "RdBu"
temp_dir = Path("temp_surf_images")


def load_imgs(paths_dict):
    return {name: nib.load(p) for name, p in paths_dict.items()}


def make_eroded_mask(mask_file, iterations):
    mimg = nib.load(mask_file)
    mask = mimg.get_fdata() > 0
    erode = binary_erosion(mask, iterations=iterations)
    return nib.Nifti1Image(erode.astype(float), mimg.affine)


def proj_mask_to_surf(mask_img, fs, hemi):
    mesh = fs.pial_left if hemi == "left" else fs.pial_right
    proj = surface.vol_to_surf(mask_img, mesh, interpolation="nearest")
    return (proj > ffa_threshold).astype(float)


def proj_vol_to_surf_t(nii4d, t, fs, hemi):
    mesh = fs.pial_left if hemi == "left" else fs.pial_right
    return surface.vol_to_surf(image.index_img(nii4d, t), mesh)


def crop_whitespace(rgb):
    # rotate & flip to match brain orientation
    img = np.fliplr(np.rot90(rgb, 3))
    # Build a mask of "not background" (tolerant white threshold)
    gray = img.mean(axis=2)
    keep = gray < 0.98
    if not np.any(keep):
        return img
    ys, xs = np.where(keep)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1, :]


def surf_figure(fs, hemi, tex, thresh, title=None):
    infl = fs.infl_left if hemi == "left" else fs.infl_right
    sulc = fs.sulc_left if hemi == "left" else fs.sulc_right
    fig = plotting.plot_surf_stat_map(
        infl, tex, hemi=hemi, view="ventral",
        threshold=thresh, cmap=cmap, bg_map=sulc,
        colorbar=False, figure=plt.figure(dpi=150)
    )
    if title:
        fig.axes[0].set_title(title, fontsize=10)
    return fig


def add_mask_contour(fs, hemi, mask_vertices, fig):
    if np.count_nonzero(mask_vertices) == 0:
        return
    infl = fs.infl_left if hemi == "left" else fs.infl_right
    try:
        plotting.plot_surf_contours(
            infl, mask_vertices, levels=[1], colors=["lime"],
            linewidths=1.5, figure=fig
        )
    except ValueError:
        # Fallback: semi-transparent fill if contours fail
        plotting.plot_surf_stat_map(
            infl, mask_vertices, hemi=hemi, view="ventral",
            threshold=0.5, cmap="Greens", alpha=0.3,
            colorbar=False, figure=fig
        )


def safe_percentile(arr, p):
    arr = np.asarray(arr)
    arr = arr[np.nonzero(arr)]
    return np.percentile(np.abs(arr), p) if arr.size else 0.0


# Main
if __name__ == "__main__":
    fs = datasets.fetch_surf_fsaverage()
    temp_dir.mkdir(exist_ok=True)

    imgs = load_imgs(data_paths)
    n_tp = imgs["Face"].shape[-1]

    ffa_mask_img = make_eroded_mask(ffa_mask_path, erosion_iter)
    ffa_surf = {
        hemi: proj_mask_to_surf(ffa_mask_img, fs, hemi)
        for hemi in ("left", "right")
    }

    # NEW LAYOUT: rows = conditions, columns = time points
    fig, axes = plt.subplots(len(imgs), n_tp, figsize=(2.5 * n_tp, 6),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})
    
    # Ensure axes is 2D even when n_tp == 1 or len(imgs) == 1
    axes = np.atleast_2d(axes)

    conditions = list(imgs.keys())
    
    for row, cond in enumerate(conditions):
        nii = imgs[cond]
        for t in range(n_tp):
            crops = []
            for hemi in ("left", "right"):
                tex = proj_vol_to_surf_t(nii, t, fs, hemi)
                thresh = safe_percentile(tex, display_percentile)
                f = surf_figure(fs, hemi, tex, thresh)
                add_mask_contour(fs, hemi, ffa_surf[hemi], f)

                tmp = temp_dir / f"{cond}_{hemi}_t{t}.png"
                plt.savefig(tmp, bbox_inches="tight")
                plt.close()

                rgb = plt.imread(tmp)[..., :3]
                crops.append(crop_whitespace(rgb))

            # make same size by center-cropping to the min h,w (simple & dependency-free)
            h = min(c.shape[0] for c in crops)
            w = min(c.shape[1] for c in crops)
            crops = [c[:h, :w] for c in crops]

            panel = np.concatenate(crops, axis=1)

            ax = axes[row, t]
            ax.imshow(panel)
            ax.axis("off")

            # Add L and R labels for left and right hemispheres
            if row == len(conditions) - 1:  # bottom row
                # Add labels below the image
                ax.text(0.25, -0.05, "L", va="top", ha="center",
                        fontsize=10, fontweight="bold", transform=ax.transAxes)
                ax.text(0.75, -0.05, "R", va="top", ha="center",
                        fontsize=10, fontweight="bold", transform=ax.transAxes)

            # Add condition label on the left
            if t == 0:
                ax.text(-0.15, 0.5, cond, va="center", ha="right",
                        fontsize=12, fontweight="bold", transform=ax.transAxes,
                        rotation=90)

            # Add time label on top
            if row == 0:
                ax.set_title(f"T={t+1}", fontsize=10, fontweight="bold")

    fig.savefig(out_png, dpi=500, bbox_inches="tight")
    plt.show()

    # clean up temp images
    for p in temp_dir.glob("*.png"):
        p.unlink()
    temp_dir.rmdir()

    print(f"Saved: {out_png}")
