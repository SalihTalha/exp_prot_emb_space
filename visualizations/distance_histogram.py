import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from calculations.utils import ensure_path_exists, get_available_tensor_files
from visualizations.utils import load_data


def get_plot(vectors, layer, model_name, dir):

    # Calculate distances from the center point for each vector in the vocabulary
    distances = [np.linalg.norm(word_vector) for word_vector in vectors]

    plt.hist(distances, bins=1000, density=False, alpha=0.75, color='b')
    plt.title(f'Count of {model_name} Vectors at layer {layer} from Center Point')
    plt.xlabel('Distance from Center Point')
    plt.ylabel('Count')
    plt.xscale('log')
    plt.savefig(f"results/distance_plots/{dir}/{model_name}/{layer}.png")
    plt.close()


def run_all():
    tensors = get_available_tensor_files()

    for t in tensors:
        model_name = t.split("_")[0]
        ensure_path_exists(f"results/distance_plots/layers/{model_name.upper()}")

        data = load_data(t)

        for i in tqdm(range(data.shape[1])):
            get_plot(data[:, i, :], i, model_name.upper(), "layers")
