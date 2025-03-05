import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import MDS
from tqdm import tqdm

from calculations.utils import load_labels, filter_by_multiple_labels
from visualizations.utils import common_labels


def plot_2d(lda_transformed, l, title="LDA Transformed Vectors in 2D", save=""):
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=lda_transformed[:, 0], y=lda_transformed[:, 1], hue=l, palette='tab10')

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Class Label')
    if save:
        plt.savefig(f"results/similarity/{save}.png")
    plt.show()


def run_all():
    files = os.listdir("results/similarity")
    for i in files:
        if i.endswith(".npy"):
            vector = np.load(i)
            plot_2d(vector)


def run_all_w_real_distance():
    distance_matrix = np.load("visualizations/distance_matrix_mean.npy")
    all_labels = load_labels()

    for label_type in tqdm(common_labels.keys()):
        indexes = filter_by_multiple_labels(all_labels, label_type, label_values=common_labels[label_type])
        if len(indexes) > 10000:
            print("More than 10k, skipping!")
            continue
        distances = ((distance_matrix[np.ix_(indexes, indexes)] - 100) * -1) / 100
        labels = list(all_labels[label_type][indexes])

        mds = MDS(n_components=2, dissimilarity='precomputed', metric=False, random_state=42)
        transformed = mds.fit_transform(distances)

        plot_2d(transformed, labels, title="MDS on ", save=f"distances/mds/{label_type}")
