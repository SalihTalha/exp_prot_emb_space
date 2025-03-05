import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

from calculations.utils import get_available_tensor_files
from visualizations.utils import ensure_path_exists, load_data


def pca(embeddings, layer, model_name):
    # Performing PCA for 2D and 3D
    pca_2d = PCA(n_components=2)

    # Transform the data
    embeddings_2d = pca_2d.fit_transform(embeddings)

    # Plotting

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], cmap='viridis')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'PCA - 2D Projection for {model_name} at Layer {layer}')

    plt.savefig(f"results/dim_reduction/layers/{model_name.upper()}/{layer}.png")



def run_all():
    tensors = get_available_tensor_files()

    for t in tensors:
        model_name = t.split("_")[0]
        ensure_path_exists(f"results/dim_reduction/layers/{model_name.upper()}")

        data = load_data(t)

        for i in tqdm(range(data.shape[1])):
            pca(data[:, i, :], i, model_name.upper())
