from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

from calculations.utils import ensure_path_exists, get_available_tensor_files
from visualizations.utils import load_data, get_seq_lens


def remove_outliers(data):
    # Q1 = np.percentile(data, 25)
    # Q3 = np.percentile(data, 75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    return [x for x in data if not (np.isnan(x) or x == 0 or np.isinf(x))]


def remove_outliers_length(data, lengths):
    """
    Removes outliers from data and ensures the corresponding elements in lengths are also removed.

    :param data: List or array of values.
    :param lengths: List or array of corresponding sequence lengths.
    :return: Filtered data and lengths arrays with outliers removed.
    """
    filtered_data = []
    filtered_lengths = []

    for d, l in zip(data, lengths):
        if not (np.isnan(d) or d == 0 or np.isinf(d)):
            filtered_data.append(d)
            filtered_lengths.append(int(l))

    return filtered_data, filtered_lengths


def get_plot(vectors, layer, model_name, dir):

    # Calculate distances from the center point for each vector in the vocabulary
    distances = [np.linalg.norm(word_vector) for word_vector in vectors]
    distances = remove_outliers(distances)

    plt.hist(distances, bins=1000, density=False, alpha=0.75, color='b')
    plt.title(f'Count of {model_name} Vectors at Layer {layer}')
    plt.xlabel('Distance from Center Point')
    plt.ylabel('Count')
    plt.xscale('log')
    plt.savefig(f"results/distance_plots/{dir}/{model_name}/{layer}.png")
    plt.close()


def get_plot_by_length(vectors, lengths, layer, model_name, dir):
    """
    Plots a single histogram of vector distances with two distributions: one for sequences with length > 300 and one for <= 300.

    :param vectors: List or array of word vectors.
    :param lengths: List or array of sequence lengths corresponding to each vector.
    :param layer: Layer number or identifier.
    :param model_name: Name of the model.
    :param dir: Directory to save the plots.
    """
    distances = np.array([np.linalg.norm(vec) for vec in vectors])
    distances, lengths = remove_outliers_length(distances, lengths)

    lengths = np.array(lengths)
    distances = np.array(distances)

    distances_long = distances[lengths > 300]
    distances_short = distances[lengths <= 300]

    plt.figure(figsize=(8, 6))

    plt.hist(distances_long, bins=1000, density=False, alpha=0.75, color='r', label='Length > 300')
    plt.hist(distances_short, bins=1000, density=False, alpha=0.75, color='b', label='Length ≤ 300')
    plt.title(f'{model_name} Vectors at Layer {layer}')
    plt.xlabel('Distance from Center Point')
    plt.ylabel('Count')
    plt.xscale('log')
    plt.legend()

    plt.savefig(f"results/distance_plots/{dir}/{model_name}/{layer}_length_based.png")
    plt.close()


def run_all():
    tensors = get_available_tensor_files()
    lengths = get_seq_lens()

    for t in tensors:
        model_name = t.split("_")[0]
        ensure_path_exists(f"results/distance_plots/layers/{model_name.upper()}")

        data = load_data(t)

        for i in tqdm(range(data.shape[1])):
            # get_plot(data[:, i, :], i, model_name.upper(), "layers")
            get_plot_by_length(data[:, i, :], lengths, i, model_name.upper(), "layers")

        del data


def length_histogram():
    return


def length_histogram_w_colors(sequence_lengths, labels_df, common_categories):
    # Assign distinct colors to each label dynamically
    palette = sns.color_palette("tab10", 4)  # Generate 4 distinct colors
    category_colors = {category: dict(zip(labels, palette)) for category, labels in common_categories.items()}

    # Define bins (same for all categories)
    num_bins = 300
    bin_edges = np.histogram_bin_edges(sequence_lengths, bins=num_bins)
    counts, _ = np.histogram(sequence_lengths, bins=bin_edges)  # Histogram based on all data

    # Create separate histograms for each category
    for category, category_labels in common_categories.items():
        # Get labels for the current category
        category_labels_data = labels_df[category].values

        # Determine dominant label per bin (without filtering sequence_lengths)
        bin_indices = np.digitize(sequence_lengths, bins=bin_edges)
        bin_colors = []

        for i in range(1, num_bins + 1):
            bin_mask = bin_indices == i
            common_label = Counter(category_labels_data[bin_mask]).most_common(1)

            if common_label:
                bin_colors.append(category_colors[category].get(common_label[0][0], "gray"))
            else:
                bin_colors.append("gray")  # Default if no data

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot histogram (use the same sequence_lengths but different colors)
        for i in range(num_bins):
            ax.bar(bin_edges[i], counts[i], width=bin_edges[i + 1] - bin_edges[i],
                   color=bin_colors[i], align="edge")

        ax.set_title(f"Category: {category.upper()}")
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Frequency")

        # Add legend for the label values
        legend_patches = [plt.Rectangle((0, 0), 1, 1, color=category_colors[category][label]) for label in
                          category_labels]
        ax.legend(legend_patches, [str(label) for label in category_labels], title="Labels", loc="upper right")

        # Show each plot separately
        plt.show()
