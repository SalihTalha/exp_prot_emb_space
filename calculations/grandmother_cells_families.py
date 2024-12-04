# import os
#
# import numpy as np
#
# from calculations.utils import load_data, load_labels
#
# labels = load_labels()
#
# def calculate(tensor):
#     return
#
#
# def run_all():
#     tensor = load_data("ankh_merged_tensor.pt")
#     res = calculate(tensor)
#     res = np.array(res)
#     np.save("results/grandmother_cells_families_ankh.npy", res)
#     del tensor
#
#     tensor = load_data("protgpt2_merged_tensor.pt")
#     res = calculate(tensor)
#     res = np.array(res)
#     np.save("results/grandmother_cells_families_protgpt2.npy", res)
#     del tensor
