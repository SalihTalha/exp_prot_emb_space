from visualizations.utils import load_int_dim, load_labels
import numpy as np


ankh_id = load_int_dim("int_dim_data_ankh.npy")
protgpt2_id = load_int_dim("int_dim_data_protgpt2.npy")

labels = load_labels()
labels = labels[:-1]


def calculate_fold_mean():
    ankh_id_means = []
    protgpt2_id_means = []
    ids = []

    for i in labels["CF"].unique():
        fil = np.array(labels[labels["CF"] == i].index)
        ankh_id_means.append(np.mean(ankh_id[fil]))
        protgpt2_id_means.append(np.mean(protgpt2_id[fil]))
        ids.append(i)

    return ids, ankh_id_means, protgpt2_id_means


def calculate_sf_mean():
    ankh_id_means = []
    protgpt2_id_means = []
    ids = []

    for i in labels["SF"].unique():
        fil = np.array(labels[labels["SF"] == i].index)
        ankh_id_means.append(np.mean(ankh_id[fil]))
        protgpt2_id_means.append(np.mean(protgpt2_id[fil]))
        ids.append(i)

    return ids, ankh_id_means, protgpt2_id_means


def calculate_f_mean():
    ankh_id_means = []
    protgpt2_id_means = []
    ids = []

    for i in labels["FA"].unique():
        fil = np.array(labels[labels["FA"] == i].index)
        ankh_id_means.append(np.mean(ankh_id[fil]))
        protgpt2_id_means.append(np.mean(protgpt2_id[fil]))
        ids.append(i)

    return ids, ankh_id_means, protgpt2_id_means


def calculate_class_mean():
    ankh_id_means = []
    protgpt2_id_means = []
    ids = []

    for i in labels["CL"].unique():
        fil = np.array(labels[labels["CL"] == i].index)
        ankh_id_means.append(np.mean(ankh_id[fil]))
        protgpt2_id_means.append(np.mean(protgpt2_id[fil]))
        ids.append(i)

    return ids, ankh_id_means, protgpt2_id_means


def calculate_type_mean():
    ankh_id_means = []
    protgpt2_id_means = []
    ids = []

    for i in labels["TP"].unique():
        fil = np.array(labels[labels["TP"] == i].index)
        ankh_id_means.append(np.mean(ankh_id[fil]))
        protgpt2_id_means.append(np.mean(protgpt2_id[fil]))
        ids.append(i)

    return ids, ankh_id_means, protgpt2_id_means

