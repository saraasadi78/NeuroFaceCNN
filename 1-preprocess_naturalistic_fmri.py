"""
This script preprocesses naturalistic fMRI data from the NNDb dataset:
- Averages subject-specific anatomical masks to create a group mask
- Applies a 0.99 threshold to retain common brain voxels
- Extracts 10-second fMRI segments where faces are present or absent
- Applies the group mask to reduce dimensionality
- Saves voxel-by-time matrices for deep learning

Requirements:
- FSL (for fslmaths)
- nibabel, numpy, matplotlib, pandas
- Directory structure from NNDb dataset (version 2.0)

Author: Sara Asadi   Date: 2025 Jun
"""


import os
import numpy as np
import nibabel as nib
import subprocess
from glob import glob
from pathlib import Path
import re
import random

#configuration
TR = 1
LAG = 4

BASE_DIR = Path("/NaturalisticDatabase_V2")
SAVE_DIR = Path("/fnof-project")
FACE_DIR = SAVE_DIR / "faces"
NOFACE_DIR = SAVE_DIR / "noface"
BRAIN_MASKS = sorted(glob(str(BASE_DIR / "sub-*" / "anat" / "sub-*_T1w_mask.nii.gz")))


def natural_sort(file_list):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list, key=alphanum_key)

def average_brain_masks():
    print(f"Found {len(BRAIN_MASKS)} brain mask files")
    command = "fslmaths "
    out_file = SAVE_DIR / f"{len(BRAIN_MASKS)}_subBrainMask_average.nii.gz"
    for i, mask in enumerate(BRAIN_MASKS):
        print(f"Processing mask {i+1}/{len(BRAIN_MASKS)}: {mask}")
        if i != len(BRAIN_MASKS) - 1:
            command += f"{mask} -add "
        else:
            command += f"{mask} -div {len(BRAIN_MASKS)} {out_file}"
    subprocess.run(command, shell=True, check=True)
    return out_file


def create_group_mask(avg_mask_file):
    data = nib.load(avg_mask_file)
    mask_data = data.get_fdata()
    mask_data = (mask_data > 0.99).astype(float)
    reshaped_mask = mask_data.reshape(-1)
    indices = np.where(reshaped_mask > 0)
    return reshaped_mask, indices


def extract_fmri_segments(task_list, reshaped_mask, indices):
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    NOFACE_DIR.mkdir(parents=True, exist_ok=True)
    
    for task in task_list:
        face_annotation_file = BASE_DIR / "stimuli" / f"stimuli-task-{task}_face-annotation.1D"
        if not face_annotation_file.exists():
            print(f"Face annotation not found for task: {task}")
            continue

        try:
            faces = np.loadtxt(face_annotation_file)
            print(f"Loaded face annotations for {task}")
        except Exception as e:
            print(f"Error reading annotations for {task}: {e}")
            continue

        fmri_files = glob(str(BASE_DIR / "sub-*" / "func" / f"sub-*_task-{task}_bold_preprocessedICA.nii.gz"))

        for fmri_file in fmri_files:
            subject_id = Path(fmri_file).parts[-3]
            sub_face_dir = FACE_DIR / subject_id
            sub_noface_dir = NOFACE_DIR / subject_id
            sub_face_dir.mkdir(parents=True, exist_ok=True)
            sub_noface_dir.mkdir(parents=True, exist_ok=True)

            try:
                data = nib.load(fmri_file).get_fdata()
            except Exception as e:
                print(f"Error loading fMRI for {subject_id}: {e}")
                continue

            valid_face_idxs = [i for i in range(len(faces)) if faces[i, 1] > 10]
            random.shuffle(valid_face_idxs)
            noface_count = sum(1 for i in range(len(faces) - 1) if faces[i+1, 0] - (faces[i, 0] + faces[i, 1]) > 10)
            valid_face_idxs = valid_face_idxs[:noface_count]

            for i in valid_face_idxs:
                index = int((faces[i, 0] / TR) + LAG)
                segment = data[:, :, :, index:index+10]
                reshaped = segment.reshape(-1, 10)[indices]
                np.savetxt(sub_face_dir / f"{subject_id}_task-{task}_face-{i}.txt", reshaped, delimiter=',')

            count = 1
            for i in range(len(faces) - 1):
                offset_time = faces[i, 0] + faces[i, 1]
                if faces[i+1, 0] - offset_time > 10:
                    index = int((offset_time / TR) + LAG)
                    segment = data[:, :, :, index:index+10]
                    reshaped = segment.reshape(-1, 10)[indices]
                    np.savetxt(sub_noface_dir / f"{subject_id}_task-{task}_noface-{count}.txt", reshaped, delimiter=',')
                    count += 1


def main():
    avg_mask_file = average_brain_masks()
    reshaped_mask, indices = create_group_mask(avg_mask_file)

    tasks = [
        '500daysofsummer', 'pulpfiction', 'theshawshankredemption', 'theprestige', 'split',
        'littlemisssunshine', '12yearsaslave', 'backtothefuture', 'citizenfour', 'theusualsuspects']
  
    extract_fmri_segments(tasks, reshaped_mask, indices)
    print("Processing complete.")


if __name__ == "__main__":
    main()
