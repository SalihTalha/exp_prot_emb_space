from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

from tqdm import tqdm

from calculations.utils import ensure_path_exists, load_data, get_available_tensor_files


def get_pca_2d_plot(vectors, layer, model, dir):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    # Ensure the directory exists
    save_dir = f"results/pca/{dir}/{model}"
    os.makedirs(save_dir, exist_ok=True)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(
        reduced_vectors[:, 0],
        reduced_vectors[:, 1],
        alpha=0.7,
        edgecolor='k',
        s=30
    )
    plt.title(f'2D PCA Plot: {model} - Layer {layer}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot
    plt.savefig(f"{save_dir}/{layer}.png")
    plt.close()


def run_all():
    tensors = get_available_tensor_files()
    for t in tensors:
        model_name = t.split("_")[0]
        data = load_data(t)

        for i in tqdm(range(data.shape[1])):
            get_pca_2d_plot(data[:, i, :], i, model_name.upper(), "layers")
