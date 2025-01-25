import numpy as np
from matplotlib import pyplot as plt
from visualizations.utils import *


def read_results():
    ankh_id = np.load("results/int_dim_data_ankh.npy")
    protgpt2_id = np.load("results/int_dim_data_protgpt2.npy")
    esm2_id = np.load("results/int_dim_data_esm2.npy")
    return ankh_id, protgpt2_id, esm2_id


ankh_res, protgpt2_res, esm2_res = read_results()
labels = load_labels()[:-1]


def run_for_label_type(label_type: str):  # label_type = TP, CL, CF, SF, FA
    arrays_ankh = []
    arrays_protgpt2 = []
    arrays_esm2 = []

    for label_value in common_labels[label_type]:
        indexes = filter_by_label(labels, label_type, label_value)
        arrays_ankh.append(ankh_res[indexes])
        arrays_protgpt2.append(protgpt2_res[indexes])
        arrays_esm2.append(esm2_res[indexes])

    create_box_plot(arrays_ankh, common_labels[label_type], "Intrinsic Dimension", f"Intrinsic Dimension using common {label_type} labels and ANKH", model_name=f"ankh_{label_type}")
    create_box_plot(arrays_protgpt2, common_labels[label_type], "Intrinsic Dimension", f"Intrinsic Dimension using common {label_type} labels and ProtGPT2", model_name=f"protgpt2_{label_type}")
    create_box_plot(arrays_esm2, common_labels[label_type], "Intrinsic Dimension", f"Intrinsic Dimension using common {label_type} labels and ESM2", model_name=f"esm2_{label_type}")


def ensure_paths():
    ensure_path_exists("results")
    ensure_path_exists("results/intrinsic_dimension")
    ensure_path_exists("results/intrinsic_dimension/data")
    ensure_path_exists("results/intrinsic_dimension/layers")


def run_all():
    ensure_paths()
    for i in common_labels.keys():
        run_for_label_type(i)


def intrinsic_dim_2nn_from_dist_matrix(D):
    """
    Compute the local 2-Nearest-Neighbors (2NN) intrinsic dimensionality
    for each point, given a distance matrix D.

    Parameters
    ----------
    D : ndarray of shape (n_samples, n_samples)
        Distance matrix where D[i, j] is the distance from point i to point j.
        We expect D[i, i] = 0 (distance to itself).

    Returns
    -------
    local_id : ndarray of shape (n_samples,)
        The 2NN intrinsic dimension estimates for each sample.
    """
    n = D.shape[0]
    local_id = np.zeros(n, dtype=float)

    for i in range(n):
        # Extract distances from point i to others
        # Ignore the diagonal (distance to itself = 0), we don't consider that a neighbor.
        row = D[i].copy()
        row[i] = np.inf  # So it doesn't get chosen as a "neighbor"

        # Sort distances in ascending order
        sorted_dists = np.sort(row)

        # The 1st neighbor distance (r1) is the smallest non-diagonal distance
        r1 = sorted_dists[0]
        # The 2nd neighbor distance (r2) is the second-smallest
        r2 = sorted_dists[1]

        # Compute the 2NN local ID = ln(2) / ln(r2 / r1)
        # Watch out for numerical issues: 0 distances, identical distances, etc.
        if r1 <= 0 or r2 <= 0 or r2 == r1:
            local_id[i] = np.nan
        else:
            local_id[i] = np.log(2.0) / np.log(r2 / r1)

    return local_id