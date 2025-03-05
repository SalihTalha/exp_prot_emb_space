import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
    plt.title(f"Intrinsic Dimension of the ANKH Model")
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
    plt.title("Intrinsic Dimension of the ProtGPT2 Model")
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
    plt.title("Intrinsic Dimension of the ESM2 Model")
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

    # plot_ankh(ankh_res)
    # plot_protgpt2(protgpt2_res)
    # plot_esm2(esm2_res)

    # Example data for intrinsic dimensions
    plt.plot(ankh_res, label="ANKH")
    plt.plot(esm2_res, label="ESM2")
    plt.plot(protgpt2_res, label="ProtGPT2")
    # Add labels and a legend
    plt.title('Intrinsic Dimensions Across Layers')
    plt.xlabel('Layers')
    plt.ylabel('Intrinsic Dimension')
    plt.legend(title='Model')
    plt.grid(True)

    # Show the plot
    plt.savefig("results/intrinsic_dimension/layers/all.png")


def plot_all_filtered():
    ankh_res, protgpt2_res, esm2_res = read_results_filtered()

    # plot_ankh(ankh_res, True)
    # plot_protgpt2(protgpt2_res, True)
    # plot_esm2(esm2_res, True)

    plt.plot(ankh_res, label="ANKH")
    plt.plot(esm2_res, label="ESM2")
    plt.plot(protgpt2_res, label="ProtGPT2")
    # Add labels and a legend
    plt.title('Intrinsic Dimensions Across Layers')
    plt.xlabel('Layers')
    plt.ylabel('Intrinsic Dimension')
    plt.legend(title='Model')
    plt.grid(True)

    # Show the plot
    plt.savefig("results/intrinsic_dimension/layers/all_filtered.png")
