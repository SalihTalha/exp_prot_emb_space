import numpy as np
from skdim.id import TwoNN
from tqdm import tqdm
from calculations.utils import load_data
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
        except:
            print("Problem occurred")
            continue

    return int_dim_data_points


def calculate_int_dim_data():
    print(os.path.exists("results"))

    tensor = load_data("ankh_merged_tensor.pt")
    res = calculate(tensor)
    res = np.array(res)
    np.save("results/int_dim_data_ankh.npy", res)
    del tensor

    tensor = load_data("protgpt2_merged_tensor.pt")
    res = calculate(tensor)
    res = np.array(res)
    np.save("results/int_dim_data_protgpt2.npy", res)
    del tensor

