import numpy as np
from matplotlib import pyplot as plt
from visualizations.utils import *


def read_results():
    ankh_id = np.load("results/int_dim_data_ankh.npy")
    protgpt2_id = np.load("results/int_dim_data_protgpt2.npy")

    return ankh_id, protgpt2_id


ankh_res, protgpt2_res = read_results()
labels = load_labels()[:-1]


def run_for_label_type(label_type: str):  # label_type = TP, CL, CF, SF, FA
    arrays_ankh = []
    arrays_protgpt2 = []

    for label_value in common_labels[label_type]:
        indexes = filter_by_label(labels, label_type, label_value)
        arrays_ankh.append(ankh_res[indexes])
        arrays_protgpt2.append(protgpt2_res[indexes])

    create_box_plot(arrays_ankh, common_labels[label_type], "Intrinsic Dimension", f"Intrinsic Dimension using common {label_type} labels and ANKH", model_name=f"ankh_{label_type}")
    create_box_plot(arrays_protgpt2, common_labels[label_type], "Intrinsic Dimension", f"Intrinsic Dimension using common {label_type} labels and ProtGPT2", model_name=f"protgpt2_{label_type}")


def run_all():
    for i in common_labels.keys():
        run_for_label_type(i)
