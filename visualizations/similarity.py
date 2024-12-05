import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def plot_2d(lda_transformed, l):
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=lda_transformed[:, 0], y=lda_transformed[:, 1], hue=l, palette='tab10')

    plt.title('LDA Transformed Vectors in 2D')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Class Label')
    plt.show()

def run_all():
    files = os.listdir("results/similarity")
    for i in files:
        if i.endswith(".npy"):
            vector = np.load(i)
            plot_2d(vector)