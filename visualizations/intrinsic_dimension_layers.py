import numpy as np
from matplotlib import pyplot as plt


def read_results():
    ankh_layers_id = np.load("results/int_dim_layers_ankh.npy")
    protgpt2_layers_id = np.load("results/int_dim_layers_protgpt2.npy")
    esm2_layers_id = np.load("results/int_dim_layers_esm2.npy")

    return ankh_layers_id, protgpt2_layers_id, esm2_layers_id


def read_results_filtered():
    ankh_layers_id = np.load("results/int_dim_filtered_layers_ankh.npy")
    protgpt2_layers_id = np.load("results/int_dim_filtered_layers_protgpt2.npy")
    esm2_layers_id = np.load("results/int_dim_filtered_layers_esm2.npy")

    return ankh_layers_id, protgpt2_layers_id, esm2_layers_id


def plot_ankh(ankh_res, filtered=False):
    # Plotting the array
    plt.plot(ankh_res)
    plt.title(f"ID through layers (ANKH)" + " filtered" if filtered else "")
    plt.xlabel("Layers")
    plt.ylabel("Intrinsic Dimension")
    plt.grid(True)
    if filtered:
        plt.savefig("results/intrinsic_dimension/layers/ankh_layers_filtered.png")
    else:
        plt.savefig("results/intrinsic_dimension/layers/ankh_layers.png")
    plt.close()


def plot_protgpt2(protgpt2_res, filtered=False):
    # Plotting the array
    plt.plot(protgpt2_res)
    plt.title("ID through layers (ProtGPT2)" + " filtered" if filtered else "")
    plt.xlabel("Layers")
    plt.ylabel("Intrinsic Dimension")
    plt.grid(True)
    if filtered:
        plt.savefig("results/intrinsic_dimension/layers/protgpt2_layers_filtered.png")
    else:
        plt.savefig("results/intrinsic_dimension/layers/protgpt2_layers.png")
    plt.close()


def plot_esm2(esm2_res, filtered=False):
    # Plotting the array
    plt.plot(esm2_res)
    plt.title("ID through layers (ESM2)" + " filtered" if filtered else "")
    plt.xlabel("Layers")
    plt.ylabel("Intrinsic Dimension")
    plt.grid(True)
    if filtered:
        plt.savefig("results/intrinsic_dimension/layers/esm2_layers_filtered.png")
    else:
        plt.savefig("results/intrinsic_dimension/layers/esm2_layers.png")
    plt.close()


def plot_all():
    ankh_res, protgpt2_res, esm2_res = read_results()

    plot_ankh(ankh_res)
    plot_protgpt2(protgpt2_res)
    plot_esm2(esm2_res)


def plot_all_filtered():
    ankh_res, protgpt2_res, esm2_res = read_results_filtered()

    plot_ankh(ankh_res, True)
    plot_protgpt2(protgpt2_res, True)
    plot_esm2(esm2_res, True)
