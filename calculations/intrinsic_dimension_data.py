import numpy as np
from skdim.id import TwoNN
from tqdm import tqdm
from calculations.utils import load_data, get_available_tensor_files
import os


def two_nn_intrinsic_dimension(data):
    intrinsic_dim = TwoNN().fit_transform(data)
    return intrinsic_dim


# def # new intrinsic dimension algo

def calculate(tensor):
    int_dim_data_points = []

    for i in tqdm(range(tensor.shape[0])):
        try:
            int_dim_data_points.append(
                two_nn_intrinsic_dimension(tensor[i, :, :])
            )
        except Exception as e:
            print("Problem occurred")
            print(e)
            continue

    return int_dim_data_points


def run_all():
    tensors = get_available_tensor_files()

    for t in tensors:
        tensor = load_data(t)
        res = calculate(tensor)
        res = np.array(res)
        np.save(f"results/int_dim_data_{t.split('_')[0]}.npy", res)
        del tensor


# def run_all_filtered(tensor):
#
#     tensors = get_available_tensor_files()
#
#     for t in tensors:
#         tensor = load_data(t)
#         indices = find_indexes_to_remove(tensor[i, :, :])
#         filtered_tensor = np.delete(tensor[:, i, :], indices, axis=0)
#         int_dim_layers.append(
#             two_nn_intrinsic_dimension(filtered_tensor)
#         )
#
#     return int_dim_layers


def run_all_filtered():
    tensors = get_available_tensor_files()
