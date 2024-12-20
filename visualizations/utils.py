import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

common_tp_labels = [1, 2, 3, 4]  # includes all data points
common_cl_labels = [1000003, 1000002, 1000001, 1000000]  # includes %94.5
common_cf_labels = [2000031, 2000014, 2000051, 2000016]  # includes %11.2
common_sf_labels = [3000038, 3000313, 3000034, 3000066]  # includes %4.8
common_fa_labels = [4000029, 4003661, 4001909, 4000229]  # includes %3.3

#
# TP
# 1    34872
# 2     1215
# 3      480
# 4      333
# Name: count, dtype: int64


# CL ->
# 1000003    9939
# 1000002    9657
# 1000001    7786
# 1000000    7491
# 1000004    2027
# Name: count, dtype: int64


# CF
# 2000031    1378
# 2000014    1094
# 2000051    1075
# 2000016     607
# 2000090     542
#            ...
# 2000190       1
# 2000961       1
# 2000182       1
# 2000958       1
# 2001469       1
# Name: count, Length: 1579, dtype: int64


# SF
# 3000038    518
# 3000313    448
# 3000034    425
# 3000066    413
# 3000135    354
#           ...
# 3000360      1
# 3001351      1
# 3000778      1
# 3000358      1
# 3001437      1
# Name: count, Length: 2816, dtype: int64


# FA
# 4000029    424
# 4003661    386
# 4001909    217
# 4000229    212
# 4000236    209
#           ...
# 4000834      1
# 4002939      1
# 4002942      1
# 4002710      1
# 4002630      1
# Name: count, Length: 5936, dtype: int64


common_labels = {
    "CL": common_cl_labels,
    "CF": common_cf_labels,
    "SF": common_sf_labels,
    "FA": common_fa_labels,
    "TP": common_tp_labels,
}

def load_int_dim(data):
    return np.load(f"results/{data}")


def load_labels():
    return pd.read_csv("labels_sf.csv")


def filter_by_label(labels: pd.DataFrame, label_name: str, label_value: str) -> list:  # return indexes
    return list(labels[labels[label_name] == label_value].index)


def create_box_plot(arrays, labels, ylabel, title, model_name):
    plt.figure(figsize=(10, 6))
    plt.boxplot(arrays, labels=labels)
    plt.ylabel(ylabel)
    plt.title(title)
    if model_name:
        plt.savefig("results/intrinsic_dimension/data/" + model_name + ".png")
    else:
        plt.show()


def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created path: {path}")
    else:
        print(f"Path already exists: {path}")


def load_data(file_name):
    return torch.load(f"./final_embeddings/{file_name}", map_location=torch.device('cpu'))
