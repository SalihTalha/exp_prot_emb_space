import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from visualizations.utils import load_labels, load_data, common_labels


def filtered_plot(vector, title: str):
    vmin = np.percentile(vector, 0.5)
    vmax = np.percentile(vector, 99.5)

    ax = plt.axes()
    sns.heatmap(vector, cmap='viridis', ax=ax, vmin=vmin, vmax=vmax)

    ax.set_title(title)
    plt.show()


def plot_two_heatmaps(vector1, vector2, title1: str, title2: str):
    """
    Plots two heatmaps side by side for the given vectors.

    Args:
        vector1 (ndarray): First 2D array for the heatmap.
        vector2 (ndarray): Second 2D array for the heatmap.
        title1 (str): Title for the first heatmap.
        title2 (str): Title for the second heatmap.
    """
    # Create subplots
    for layer in range(vector1.shape[1]):
        v1 = vector1[:, layer, :]
        v2 = vector2[:, layer, :]
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Calculate vmin and vmax for each vector
        mean_v1 = torch.mean(v1, dim=0)
        mean_v2 = torch.mean(v2, dim=0)
        vmin1, vmax1 = np.percentile(mean_v1, 0.5), np.percentile(mean_v1, 99.5)
        vmin2, vmax2 = np.percentile(mean_v2, 0.5), np.percentile(mean_v2, 99.5)

        # Plot the first heatmap
        sns.heatmap(v1, cmap='viridis', ax=axs[0], vmin=vmin1, vmax=vmax1)
        axs[0].set_title(title1)

        # Plot the second heatmap
        sns.heatmap(v2, cmap='viridis', ax=axs[1], vmin=vmin2, vmax=vmax2)
        axs[1].set_title(title2)

        # Adjust layout and display
        plt.tight_layout()
        plt.savefig('foo.png', bbox_inches='tight')


def plot_8_heatmaps(tensor, indexes, titles, filename):
    """
    Plots 8 heatmaps in a grid with 4 rows and 2 columns.

    Args:
        vectors (list of ndarray): List of 8 2D arrays for the heatmaps.
        titles (list of str): List of 8 titles for each heatmap.
    """
    if len(indexes) != 8 or len(titles) != 8:
        raise ValueError("You must provide exactly 8 vectors and 8 titles.")

    for layer in tqdm(range(tensor[indexes[0]].shape[1])):
        # Create a 4x2 grid for the plots
        fig, axs = plt.subplots(4, 2, figsize=(14, 20))
        axs = axs.flatten()  # Flatten the axes for easy indexing

        # Plot each heatmap
        for i, (index, title) in enumerate(zip(indexes, titles)):
            vmin = np.percentile(tensor[index][layer], 0.5)
            vmax = np.percentile(tensor[index][layer], 99.5)

            sns.heatmap(tensor[index][layer], cmap='viridis', ax=axs[i], vmin=vmin, vmax=vmax)
            axs[i].set_title(title)

        plt.title = f"Layer: {layer}"
        # Adjust layout and display
        plt.tight_layout()
        plt.savefig(filename)
        # plt.show()


def filter_common_labels(tensor, labels, label_name, label_value):
    indexes = np.array(list(labels[labels[label_name] == label_value].index))

    filtered_tensor = tensor[indexes]

    mask = torch.ones(tensor.size(0), dtype=torch.bool)  # Initialize all True

    mask[indexes] = False  # Set False where you want to remove
    # Select only the rows you want to keep
    filtered_negative_tensor = tensor[mask]

    return filtered_tensor, filtered_negative_tensor


def run_all():
    labels = load_labels()

    tensor = load_data("ankh_merged_tensor.pt")

    for i in tqdm(list(common_labels.keys())[-2:-1]):
        indexes, titles = [], []
        for j in tqdm(common_labels[i]):
            label_name = i
            label_value = j
            positive_indexes = np.array(list(labels[labels[label_name] == label_value].index))
            negative_indexes = torch.ones(tensor.size(0), dtype=torch.bool)  # Initialize all True
            negative_indexes[positive_indexes] = False  # Set False where you want to remove

            indexes.append(positive_indexes)
            titles.append(f"ANKH Activations for Label ({i}: {j})")
            indexes.append(negative_indexes)
            titles.append("Negative")

        plot_8_heatmaps(tensor, indexes, titles, f"results/grandmother_cells/ankh_{i}_gmcells.png")

    del tensor

    tensor = load_data("protgpt2_merged_tensor.pt")

    for i in tqdm(common_labels.keys()):
        indexes, titles = [], []
        for j in tqdm(common_labels[i]):
            label_name = i
            label_value = j
            positive_indexes = np.array(list(labels[labels[label_name] == label_value].index))
            negative_indexes = torch.ones(tensor.size(0), dtype=torch.bool)  # Initialize all True
            negative_indexes[positive_indexes] = False  # Set False where you want to remove

            indexes.append(positive_indexes)
            titles.append(f"ProtGPT2 Activations for Label ({i}: {j})")
            indexes.append(negative_indexes)
            titles.append("Negative")

        plot_8_heatmaps(tensor, indexes, titles, f"results/grandmother_cells/protgpt2_{i}_gmcells.png")
