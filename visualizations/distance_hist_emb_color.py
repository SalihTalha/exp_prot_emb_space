from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from calculations.utils import get_available_tensor_files
from visualizations.utils import load_labels, ensure_path_exists, load_data

labels = load_labels()

cf_labels = labels["CF"]


def get_plot(vectors, labels, model_name="", num_bins=100, fixed=False, label_map={}, update_colors={}):
    # Set the center point (e.g., point 0)
    center_point = np.zeros_like(vectors[0])

    if label_map:
        labels = [label_map[label] for label in labels]

    # Calculate distances from the center point for each vector
    distances = [np.linalg.norm(word_vector - center_point) for word_vector in vectors]

    # Create a histogram and get bin edges
    counts, bin_edges = np.histogram(distances, bins=num_bins, density=True)

    # Assign labels to bins
    bin_labels = [[] for _ in range(num_bins)]
    for distance, label in zip(distances, labels):
        bin_index = np.digitize(distance, bin_edges) - 1
        bin_index = min(bin_index, num_bins - 1)  # Ensure the bin_index doesn't exceed the last bin
        bin_labels[bin_index].append(label)

    # Determine the most frequent label per bin, skipping 'nan'
    dominant_labels = []
    for bin_label in bin_labels:
        if bin_label:
            sorted_labels = Counter(bin_label).most_common()
            most_common_label = next((label for label, _ in sorted_labels if label != 'nan'), None)
            dominant_labels.append(most_common_label)
        else:
            dominant_labels.append(None)

    # Counting the frequency of each label in dominant_labels, excluding 'nan'
    label_frequency = Counter(filter(lambda x: x is not None and x != 'nan', dominant_labels))

    # Identifying the top five most frequent labels
    top_labels = [label for label, _ in label_frequency.most_common(5)]

    # Define a list of distinct colors for the top labels
    distinct_colors = ['red', 'green', 'blue', 'purple', 'orange']

    # Assign distinct colors to the top five labels
    label_color_map = {label: color for label, color in zip(top_labels, distinct_colors)}

    # Use the default color map for the remaining labels
    unique_labels = list(set(filter(lambda x: x is not None and x != 'nan' and x not in top_labels, dominant_labels)))
    default_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_color_map.update(dict(zip(unique_labels, default_colors)))

    if update_colors:
        label_color_map.update(update_colors)

    # Creating a legend mapping
    legend_handles = [plt.Line2D([0], [0], color=label_color_map[label], lw=4, label=label) for label in top_labels]

    # Coloring each bin
    for i in range(len(counts)):
        bin_color = label_color_map.get(dominant_labels[i], 'grey')
        plt.fill_betweenx([0, counts[i]], bin_edges[i], bin_edges[i + 1], color=bin_color, alpha=0.75)

    # Plotting
    plt.title(f'Density of {model_name.upper()} Vectors from Center Point')
    plt.xlabel('Distance from Center Point')
    plt.ylabel('Density')
    if fixed:
        plt.xlim(fixed[0], fixed[1])
    plt.legend(handles=legend_handles, title="Top 5 Labels")
    plt.savefig(f"results/distance_plots_emb/{model_name}")


def run_all():
    tensors = get_available_tensor_files()

    tensors = ["esm2_merged_tensor.pt", "ankh_merged_tensor.pt", "protgpt2_merged_tensor.pt"]
    ensure_path_exists(f"results/distance_plots_emb/layers/ESM2")
    data = load_data(tensors[0])
    get_plot(data[:, 37, :], cf_labels, "ESM2", 300)
    del data

    ensure_path_exists(f"results/distance_plots_emb/layers/ANKH")
    data = load_data(tensors[1])
    get_plot(data[:, 37, :], cf_labels, "ANKH", 300, fixed=(2000000,4000000))
    del data

    ensure_path_exists(f"results/distance_plots_emb/layers/PROTGPT2")
    data = load_data(tensors[2])
    get_plot(data[:, 30, :], cf_labels, "PROTGPT2", 1000, fixed=(3000, 7000))
    del data
    #
    # for t in tensors:
    #     model_name = t.split("_")[0]
    #     ensure_path_exists(f"results/distance_plots_emb/layers/{model_name.upper()}")
    #
    #     data = load_data(t)
    #
    #     # for i in tqdm(range(data.shape[1])):
    #     #     # get_plot(data[:, i, :], i, model_name.upper(), "layers")
    #     get_plot(data[:, 30, :], cf_labels, model_name, 300)
    #
    #     del data

