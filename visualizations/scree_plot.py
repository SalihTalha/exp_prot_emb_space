import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from calculations.utils import get_available_tensor_files
from visualizations.utils import load_data, ensure_path_exists

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def get_scree_plot(vector, layer, model, dir):
    # Perform PCA
    pca = PCA()
    pca.fit(vector)
    explained_variance_ratio = pca.explained_variance_ratio_ * 100
    cumulative_sum = np.cumsum(explained_variance_ratio)
    explained_variance_ratio = explained_variance_ratio[:100]

    # Identify the index of the component that crosses 90% cumulative variance
    index = np.argmax(cumulative_sum > 90)

    # Create a list of colors for the bars
    colors = ['red' if i < index else 'blue' for i in range(len(explained_variance_ratio))]

    # Ensure the directory exists
    save_dir = f"results/scree_plots/{dir}/{model}"
    os.makedirs(save_dir, exist_ok=True)

    # Plot the scree plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color=colors)
    plt.xlabel('Principal Components')
    plt.ylabel('Percentage of Variance Explained')
    plt.title(f'Scree Plot: {model} - Layer {layer} - ID {index}')
    plt.axvline(x=index, color='red', linestyle='--',
                label=f'90% Variance Threshold (PC {index})')  # Optional: Add a vertical line
    plt.legend()

    # Save the plot
    plt.savefig(f"{save_dir}/{layer}.png")
    plt.close()


def run_all():
    tensors = get_available_tensor_files()

    for t in tensors:
        tensor = load_data(t)
        model_name = t.split("_")[0]
        ensure_path_exists(f"results/scree_plots/layers/{model_name.upper()}")

        for i in tqdm(range(tensor.shape[1])):
            get_scree_plot(tensor[:, i, :], i, model_name.upper(), "layers")

