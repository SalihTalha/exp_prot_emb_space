import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from calculations.utils import get_available_tensor_files
from visualizations.utils import common_labels, filter_by_label, load_labels, ensure_path_exists, load_data


def pca(embeddings, layer, model_name, labels_df):
    plt.figure(figsize=(8, 6))
    for category, label_values in common_labels.items():
        filtered_embeddings = []
        lens = []
        for label_value in label_values:
            indexes = filter_by_label(labels_df, category, label_value)
            filtered_embeddings.extend(embeddings[indexes])
            lens.append(len(indexes))

        pca_2d = PCA(n_components=2)
        embeddings_2d = pca_2d.fit_transform(filtered_embeddings)

        j_0 = 0
        j_1 = 0
        for index, i in enumerate(lens):
            j_1 += i
            plt.scatter(embeddings_2d[j_0:j_1, 0], embeddings_2d[j_0:j_1, 1], label=label_values[index], alpha=0.6)
            j_0 = j_1

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'PCA - 2D Projection for {model_name} at Layer {layer} for Category {category}')
        plt.legend()

        plt.savefig(f"results/dim_reduction/layers/{model_name.upper()}/{category}_{layer}_colored.png")
        plt.close()


def run_all():
    tensors = get_available_tensor_files()
    labels_df = load_labels()  # Load labels once

    for t in tensors:
        model_name = t.split("_")[0]
        ensure_path_exists(f"results/dim_reduction/layers/{model_name.upper()}")

        data = load_data(t)

        for i in tqdm(range(data.shape[1])):
            pca(data[:, i, :], i, model_name.upper(), labels_df)
