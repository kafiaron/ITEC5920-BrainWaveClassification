import numpy as np
from scipy.io import loadmat
import os

SUBJECTS = ["aa", "al", "av", "aw", "ay"]

def load_subject(data_folder, subj):
    feature_path = os.path.join(data_folder, f"features_{subj}_440_0.mat")
    label_path   = os.path.join(data_folder, f"true_labels_{subj}.mat")

    # Load the EEG feature data.
    # The file stores it in the shape (channels, time_points, trials),
    # which for this dataset is (6, 201, 280).
    X_raw = loadmat(feature_path)["features"]

    # Load the labels for each trial.
    # Depending on how MATLAB saved the file, this may be (280,) or (1, 280).
    y_raw = loadmat(label_path)["true_y"]

    # Rearrange the EEG data so that each trial is the first dimension.
    # After this step the shape becomes (trials, time_points, channels),
    # which is (280, 201, 6) for this dataset.
    X = np.transpose(X_raw, (2, 1, 0))

    # Convert labels into a simple 1‑D array.
    y = y_raw.flatten()

    return X, y


def build_dataset(data_folder):
    X_list = []
    y_list = []

    # Load each subject’s data and store it.
    for subj in SUBJECTS:
        X, y = load_subject(data_folder, subj)
        X_list.append(X)
        y_list.append(y)

    # Combine all subjects into one dataset.
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y
