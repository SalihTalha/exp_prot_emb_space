import os

import pandas as pd
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


common_tp_labels = [1, 2, 3, 4]  # includes all data points
common_cl_labels = [1000003, 1000002, 1000001, 1000000]  # includes %94.5
common_cf_labels = [2000031, 2000014, 2000051, 2000016]  # includes %11.2
common_sf_labels = [3000038, 3000313, 3000034, 3000066]  # includes %4.8
common_fa_labels = [4000029, 4003661, 4001909, 4000229]  # includes %3.3

common_labels = {
    "CL": common_cl_labels,  # class-level
    "CF": common_cf_labels,  # class-fold
    "SF": common_sf_labels,  # superfamily
    "FA": common_fa_labels,  # family
    "TP": common_tp_labels,  # type
}


def load_data(file_name):
    return torch.load(f"./final_embeddings/{file_name}", map_location=torch.device('cpu'))


def load_int_dim(data):
    return np.load(f"results/{data}")


def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created path: {path}")
    else:
        print(f"Path already exists: {path}")


def load_labels():
    return pd.read_csv("labels_sf.csv")

def get_available_tensor_files():
    # return os.listdir("final_embeddings")
    return ["esm2_merged_tensor.pt"]  # , "ankh_merged_tensor.pt", "protgpt2_merged_tensor.pt"

def filter_by_label(labels: pd.DataFrame, label_name: str, label_value: str) -> list:  # return indexes
    return list(labels[labels[label_name] == label_value].index)


def filter_by_multiple_labels(labels: pd.DataFrame, label_name: str, label_values: list) -> list:
    return list(labels[labels[label_name].isin(label_values)].index)


def find_indexes_to_remove(data):
    """
    Computes r2/r1 using the Nearest Neighbors distances.
    Identifies indices of invalid values, outliers, and the last 500
    points in the ratio-sorted distribution.

    Returns:
        indices_to_remove (np.ndarray): array of indices to remove from 'data'.
    """

    # Fit k-NN and get distances
    knn = NearestNeighbors(n_neighbors=3).fit(data)
    distances, _ = knn.kneighbors(data)

    r1 = distances[:, 1]  # distance to the 1st neighbor
    r2 = distances[:, 2]  # distance to the 2nd neighbor
    ratios = r2 / r1  # ratio = r2/r1

    # ------------------------------------------------------------------
    # 1) Identify invalid (NaN, Inf) ratio values
    # ------------------------------------------------------------------
    invalid_mask = np.isnan(ratios) | np.isinf(ratios)

    # ------------------------------------------------------------------
    # 2) Identify outliers based on IQR
    #    Only compute IQR on valid (non-NaN, non-Inf) points
    # ------------------------------------------------------------------
    valid_ratios = ratios[~invalid_mask]
    Q1 = np.percentile(valid_ratios, 25)
    Q3 = np.percentile(valid_ratios, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_mask = (ratios < lower_bound) | (ratios > upper_bound)

    # ------------------------------------------------------------------
    # Combine invalid + outlier masks
    # ------------------------------------------------------------------
    # We'll keep only those that are both valid and in-bounds
    valid_and_in_bounds_mask = ~invalid_mask & ~outlier_mask

    # ------------------------------------------------------------------
    # 3) Identify indices for the last 500 ratios among valid in-bounds
    # ------------------------------------------------------------------
    # Sort the "valid and in-bounds" points by their ratio value
    valid_in_bounds_indices = np.where(valid_and_in_bounds_mask)[0]
    sorted_valid_in_bounds_indices = valid_in_bounds_indices[np.argsort(ratios[valid_in_bounds_indices])]

    # We only remove the last 500 if we have more than 500 valid points
    if len(sorted_valid_in_bounds_indices) > 500:
        last_500_indices = sorted_valid_in_bounds_indices[-500:]
    else:
        last_500_indices = np.array([], dtype=int)

    # ------------------------------------------------------------------
    # Gather all indices to remove:
    #   - invalid ratio points
    #   - outliers
    #   - last 500 points in the sorted distribution
    # ------------------------------------------------------------------
    invalid_or_outlier_indices = np.where(invalid_mask | outlier_mask)[0]
    indices_to_remove = np.concatenate([invalid_or_outlier_indices, last_500_indices])
    indices_to_remove = np.unique(indices_to_remove)  # ensure sorted & unique

    return indices_to_remove
