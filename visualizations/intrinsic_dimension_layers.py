import numpy as np
from matplotlib import pyplot as plt


def read_results():
    ankh_layers_id = np.load("results/int_dim_layers_ankh.npy")
    protgpt2_layers_id = np.load("results/int_dim_layers_protgpt2.npy")

    return ankh_layers_id, protgpt2_layers_id


ankh_res, protgpt2_res = read_results()


def plot_ankh():
    # Plotting the array
    plt.plot(ankh_res)
    plt.title("ID through layers (ANKH)")
    plt.xlabel("Layers")
    plt.ylabel("Intrinsic Dimension")
    plt.grid(True)
    plt.savefig("results/intrinsic_dimension/layers/ankh_layers.png")
    plt.close()

def plot_protgpt2():
    # Plotting the array
    plt.plot(protgpt2_res)
    plt.title("ID through layers (ProtGPT2)")
    plt.xlabel("Layers")
    plt.ylabel("Intrinsic Dimension")
    plt.grid(True)
    plt.savefig("results/intrinsic_dimension/layers/protgpt2_layers.png")
    plt.close()


def plot_both():
    plot_ankh()
    plot_protgpt2()

