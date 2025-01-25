import os

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from matplotlib import pyplot as plt

common_tp_labels = [1, 2, 3, 4]  # includes all data points
common_cl_labels = [1000003, 1000002, 1000001, 1000000]  # includes %94.5
common_cf_labels = [2000031, 2000014, 2000051, 2000016]  # includes %11.2
common_sf_labels = [3000038, 3000313, 3000034, 3000066]  # includes %4.8
common_fa_labels = [4000029, 4003661, 4001909, 4000229]  # includes %3.3

# Type
# Class
# Class-fold
# Super family
# Family

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
    "CL": common_cl_labels,  # Class
    "CF": common_cf_labels,  # Class-fold
    "SF": common_sf_labels,  # Super family
    "FA": common_fa_labels,  # Family
    "TP": common_tp_labels,  # Type
}


def parse_txt_to_dict(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Skip lines that start with '#' or are empty
            if line.startswith('#') or not line.strip():
                continue

            # Split the line into NODE_ID and NODE_NAME
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                node_id, node_name = parts
                data_dict[int(node_id)] = node_name.strip()

    return data_dict

# Example usage:
file_path = "scop-des-latest.txt"  # Replace with the path to your text file
data_dict = parse_txt_to_dict(file_path)

def load_int_dim(data):
    return np.load(f"results/{data}")


def load_labels():
    return pd.read_csv("labels_sf.csv")


def get_seq_lens():
    df = load_labels()
    fasta_file = "scop_sf_represeq_lib_latest.fa.txt"
    seq_records = list(SeqIO.parse(fasta_file, "fasta"))

    assert len(seq_records) == len(df), "Number of FASTA sequences != number of rows in CSV!"

    # Extract protein lengths (index aligns with df rows)
    seq_lengths = [len(record.seq) for record in seq_records]
    return seq_lengths


def get_label_name(label_id):
    return data_dict.get(label_id, "")

def filter_by_label(labels: pd.DataFrame, label_name: str, label_value: str) -> list:  # return indexes
    return list(labels[labels[label_name] == label_value].index)


def create_box_plot(arrays, labels, ylabel, title, model_name, save_dir="results/intrinsic_dimension/data/", ylim=()):
    plt.figure(figsize=(10, 6))
    plt.boxplot(arrays, labels=labels)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(ylim[0], ylim[1])  # Replace `min_value` and `max_value` with your desired limits

    if model_name:
        plt.savefig(save_dir + model_name + ".png")
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
