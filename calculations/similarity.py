import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import silhouette_score
import glob

from calculations.utils import load_data, load_labels, common_labels, filter_by_multiple_labels, ensure_path_exists, \
    get_available_tensor_files

try:
    os.mkdir("results/similarity")
except Exception as e:
    print(e)


def calculate_lda(vector, labels, model_name, label_type, layer_number):
    lda = LDA(n_components=2)  # You can set n_components to the number of classes - 1 if it's known

    # Fit LDA to your data
    lda.fit(vector, labels)

    # Transform the embeddings to the lower-dimensional space
    lda_transformed = lda.transform(vector)

    np.save(f"results/similarity/{model_name}/vectors/{label_type}_{layer_number}.npy", lda_transformed)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=lda_transformed[:, 0], y=lda_transformed[:, 1], hue=labels, palette='tab10')

    plt.title(f'LDA with {label_type} labels on {model_name} layer {layer_number}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title=f"{label_type} Label")
    plt.savefig(f"results/similarity/{model_name}/{label_type}/{layer_number}")
    plt.close()

    return lda_transformed


def ensure_paths(model_name):
    ensure_path_exists("results/similarity/accuracy_plots")
    ensure_path_exists(f"results/similarity/{model_name}")
    ensure_path_exists(f"results/similarity/{model_name}/TP")
    ensure_path_exists(f"results/similarity/{model_name}/SF")
    ensure_path_exists(f"results/similarity/{model_name}/CL")
    ensure_path_exists(f"results/similarity/{model_name}/CF")
    ensure_path_exists(f"results/similarity/{model_name}/FA")
    ensure_path_exists(f"results/similarity/{model_name}/vectors")


# Define a function to plot silhouette scores from a CSV file
def plot_silhouette_score(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Extract layer and silhouette_score
    layers = df['layer']
    scores = df['silhouette_score']

    # Return the data for plotting
    return layers, scores


def plot_accuracies():
    # Find all CSV files that start with each model's prefix and group by label
    models_prefixes = {
        "ANKH": "acc_ANKH",
        "ESM2": "acc_ESM2",
        "PROTGPT2": "acc_PROTGPT2"
    }

    # Define the possible labels
    labels = ['CF', 'CL', 'FA', 'SF', 'TP']

    # Loop through the model prefixes to find and plot their files
    for model_name, prefix in models_prefixes.items():
        # Create a new figure for each model
        plt.figure(figsize=(10, 6))

        # Loop through labels to group CSV files
        for label in labels:
            # Find all CSV files for the current model and label
            csv_files = glob.glob(f"./results/similarity/accuracy_plots/{prefix}_{label}_*.csv")

            # Loop through the files for the current label
            for csv_file in csv_files:
                # Get the silhouette score data
                layers, scores = plot_silhouette_score(csv_file)

                # Plot the silhouette score for the label
                plt.plot(layers, scores, marker='o', label=f'{label} ({csv_file.split("_")[-1].split(".")[0]})')

        # Customize and show the plot for the current model
        plt.title(f'Silhouette Scores for {model_name}')
        plt.xlabel('Layer')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"./results/similarity/accuracy_plots/accuracies_total_{model_name}.png")


def run_all():
    all_labels = load_labels()

    tensors = get_available_tensor_files()
    for t in tensors:
        model_name = t.split("_")[0].upper()
        ensure_paths(model_name)
        embeddings = load_data(t)
        for label_type in tqdm(common_labels.keys()):
            indexes = filter_by_multiple_labels(all_labels, label_type, label_values=common_labels[label_type])
            if os.path.exists(f"results/similarity/accuracy_plots/acc_{model_name}_{label_type}_{len(indexes)}.csv"):
                df = pd.read_csv(f"results/similarity/accuracy_plots/acc_{model_name}_{label_type}_{len(indexes)}.csv")
            else:
                results = []
                embs = embeddings[indexes]
                labels = list(all_labels[label_type][indexes])
                for layer in tqdm(range(embs.shape[1])):
                    try:
                        layer_embs = embs[:, layer, :]
                        if os.path.exists(f"results/similarity/{model_name}/vectors/{label_type}_{layer}.npy"):
                            transformed = np.load(f"results/similarity/{model_name}/vectors/{label_type}_{layer}.npy")
                        else:
                            transformed = calculate_lda(layer_embs, labels, model_name, label_type, layer)

                        # tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
                        # embeddings_2d = tsne.fit_transform(transformed)
                        score = silhouette_score(transformed, labels)

                        # X_train, X_test, y_train, y_test = train_test_split(transformed, labels, test_size=0.2, random_state=42)
                        #
                        # # Train KNN classifier
                        # knn = KNeighborsClassifier(n_neighbors=3)
                        # knn.fit(X_train, y_train)

                        # Predict on the test set
                        # y_pred = knn.predict(X_test)
                        results.append({
                            "layer": layer,
                            "silhouette_score": score,
                            # "accuracy_score": accuracy_score(y_test, y_pred),
                            # "precision_score": precision_score(y_test, y_pred, average='weighted'),
                            # "recall_score": recall_score(y_test, y_pred, average='weighted'),
                            # "f1_score": f1_score(y_test, y_pred, average='weighted'),
                            # "lda_score": None # TODO
                        })
                    except Exception as e:
                        print(e)
                        continue

                df = pd.DataFrame(results)
                df.to_csv(f"results/similarity/accuracy_plots/acc_{model_name}_{label_type}_{len(indexes)}.csv")

            layers = list(df["layer"])

            # Create the plot
            plt.figure(figsize=(10, 6))

            # Plot each metric
            plt.plot(layers, list(df["silhouette_score"]), label='Accuracy', marker='^')
            # plt.plot(layers, list(df["accuracy_score"]), label='Accuracy', marker='^')
            # plt.plot(layers, list(df["precision_score"]), label='Precision', marker='o')
            # plt.plot(layers, list(df["recall_score"]), label='Recall', marker='s')
            # plt.plot(layers, list(df["f1_score"]), label='F1 Score', marker='d')

            # Adding labels, title, and grid
            plt.title('Layer-wise Metrics')
            plt.xlabel('Layer Number')
            plt.ylabel('Metric Value')
            plt.ylim(0, 1)  # Assuming metrics range from 0 to 1
            plt.xticks(layers)  # Show all layer numbers on the x-axis
            plt.grid(True)
            # plt.legend()
            plt.tight_layout()

            # Save the plot instead of showing it
            output_path = f"results/similarity/accuracy_plots/accuracies_{model_name}_{label_type}.png"
            plt.savefig(output_path)  # Save with high resolution
            plt.close()  # Close the plot to free memory
