import os

import numpy as np
import pandas as pd
import torch

common_tp_labels = [1, 2, 3, 4]  # includes all data points
common_cl_labels = [1000003, 1000002, 1000001, 1000000]  # includes %94.5
common_cf_labels = [2000031, 2000014, 2000051, 2000016]  # includes %11.2
common_sf_labels = [3000038, 3000313, 3000034, 3000066]  # includes %4.8
common_fa_labels = [4000029, 4003661, 4001909, 4000229]  # includes %3.3

common_labels = {
    "TP": common_tp_labels,
    "CL": common_cl_labels,
    "CF": common_cf_labels,
    "SF": common_sf_labels,
    "FA": common_fa_labels,
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


def filter_by_label(labels: pd.DataFrame, label_name: str, label_value: str) -> list:  # return indexes
    return list(labels[labels[label_name] == label_value].index)


def filter_by_multiple_labels(labels: pd.DataFrame, label_name: str, label_values: list) -> list:
    return list(labels[labels[label_name].isin(label_values)].index)
