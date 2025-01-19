import numpy as np
from skdim.id import TwoNN
from tqdm import tqdm
from calculations.utils import load_data, find_indexes_to_remove, get_available_tensor_files
import os


def two_nn_intrinsic_dimension(data):
    intrinsic_dim = TwoNN().fit_transform(data)
    return intrinsic_dim



def calculate(tensor):
    int_dim_layers = []

    for i in tqdm(range(tensor.shape[1])):
        int_dim_layers.append(
            two_nn_intrinsic_dimension(tensor[:, i, :])
        )

    return int_dim_layers


def calculate_filtered(tensor):
    int_dim_layers = []

    ### Calculate ID
    for i in tqdm(range(tensor.shape[1])):
        indices = find_indexes_to_remove(tensor[:, i, :])
        filtered_tensor = np.delete(tensor[:, i, :], indices, axis=0)
        int_dim_layers.append(
            two_nn_intrinsic_dimension(filtered_tensor)
        )

    return int_dim_layers


def run_all():
    print(os.path.exists("results"))
    # tensor = load_data("ankh_merged_tensor.pt")
    # res = calculate(tensor)
    # res = np.array(res)
    # np.save("results/int_dim_layers_ankh.npy", res)
    # del tensor
    #
    # tensor = load_data("protgpt2_merged_tensor.pt")
    # res = calculate(tensor)
    # res = np.array(res)
    # np.save("results/int_dim_layers_protgpt2.npy", res)
    # del tensor

    tensor = load_data("esm2_merged_tensor.pt")
    res = calculate_filtered(tensor)
    res = np.array(res)
    np.save("results/int_dim_filtered_layers_esm2.npy", res)
    del tensor

    tensor = load_data("ankh_merged_tensor.pt")
    res = calculate_filtered(tensor)
    res = np.array(res)
    np.save("results/int_dim_filtered_layers_ankh.npy", res)
    del tensor

    tensor = load_data("protgpt2_merged_tensor.pt")
    res = calculate_filtered(tensor)
    res = np.array(res)
    np.save("results/int_dim_filtered_layers_protgpt2.npy", res)
    del tensor
