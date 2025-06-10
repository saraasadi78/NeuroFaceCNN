import os
import numpy as np
import tensorflow as tf 
import pickle
import logging

#logging set up
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#base directory (change this path as needed)
BASE_PATH = r"C:\Users\sara.asadi\Desktop\pypro"

#subfolders
FACE_FOLDER = "face"
NOFACE_FOLDER = "noface"

#counting the number of .txt files in all subject subfolders
def count_text_files(folder_name):
    folder_path = os.path.join(BASE_PATH, folder_name)
    count = sum(
        file.endswith('.txt')
        for subject in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, subject))
        for file in os.listdir(os.path.join(folder_path, subject))
    )
    return count


#load .txt data from a specified folder and return as a list of 2D NumPy arrays
def load_text_data(folder_name, limit=None):
    
    data = []
    subject_path = os.path.join(BASE_PATH, folder_name)
    subject_folders = sorted(os.listdir(subject_path))
    for idx, subject in enumerate(subject_folders):
        if limit is not None and idx >= limit:
            break
        full_subject_path = os.path.join(subject_path, subject)
        if os.path.isdir(full_subject_path):
            for file in sorted(os.listdir(full_subject_path)):
                if file.endswith(".txt"):
                    file_path = os.path.join(full_subject_path, file)
                    matrix = np.loadtxt(file_path, delimiter=",")
                    data.append(matrix)
            logging.info(f"Loaded subject: {subject} ({folder_name})")
    logging.info(f"Processed {min(len(subject_folders), limit or len(subject_folders))} subjects from '{folder_name}'.")
    return data


def main():
    logging.info("Counting input files...")
    face_count = count_text_files(FACE_FOLDER)
    noface_count = count_text_files(NOFACE_FOLDER)
    logging.info(f"Found {face_count} '.txt' files in '{FACE_FOLDER}'")
    logging.info(f"Found {noface_count} '.txt' files in '{NOFACE_FOLDER}'")

    logging.info("Loading data...")
    face_data = load_text_data(FACE_FOLDER)
    noface_data = load_text_data(NOFACE_FOLDER)

    logging.info("Constructing input arrays...")
    X = np.array(face_data + noface_data)
    #label
    y = np.array([1] * len(face_data) + [0] * len(noface_data))

    #reshape for CNN input: (samples, height, width, channels)
    height, width = X.shape[1], X.shape[2]
    X = X.reshape(-1, height, width, 1)

    logging.info(f"X shape: {X.shape}, y shape: {y.shape}, X size: {X.nbytes / 1e6:.2f} MB")

    output_path = os.path.join(BASE_PATH, "fmri_data.pickle")
    logging.info(f"Saving preprocessed data to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
    logging.info("Data successfully saved.")


if __name__ == "__main__":
    main()

#3606 files, 41489 rows, 10 timepoint, 1 channel 
#face: 1 / noface: 0
