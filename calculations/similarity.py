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

from calculations.utils import load_data, load_labels, common_labels, filter_by_multiple_labels, ensure_path_exists

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


def ensure_paths():
    ensure_path_exists("results")
    ensure_path_exists("results/similarity")
    ensure_path_exists("results/similarity/accuracy_plots")
    ensure_path_exists("results/similarity/ankh")
    ensure_path_exists("results/similarity/ankh/TP")
    ensure_path_exists("results/similarity/ankh/SF")
    ensure_path_exists("results/similarity/ankh/CL")
    ensure_path_exists("results/similarity/ankh/CF")
    ensure_path_exists("results/similarity/ankh/FA")
    ensure_path_exists("results/similarity/ankh/vectors")
    ensure_path_exists("results/similarity/protgpt2")
    ensure_path_exists("results/similarity/protgpt2/TP")
    ensure_path_exists("results/similarity/protgpt2/SF")
    ensure_path_exists("results/similarity/protgpt2/CL")
    ensure_path_exists("results/similarity/protgpt2/CF")
    ensure_path_exists("results/similarity/protgpt2/FA")
    ensure_path_exists("results/similarity/protgpt2/vectors")


def run_all():
    ensure_paths()

    all_labels = load_labels()

    ankh_embs = load_data("ankh_merged_tensor.pt")
    for label_type in tqdm(common_labels.keys()):
        indexes = filter_by_multiple_labels(all_labels, label_type, label_values=common_labels[label_type])
        results = []
        embs = ankh_embs[indexes]
        labels = list(all_labels[label_type][indexes])
        for layer in tqdm(range(embs.shape[1])):
            layer_embs = embs[:, layer, :]
            transformed = calculate_lda(layer_embs, labels, "ankh", label_type, layer)

            X_train, X_test, y_train, y_test = train_test_split(transformed, labels, test_size=0.2, random_state=42)

            # Train KNN classifier
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)

            # Predict on the test set
            y_pred = knn.predict(X_test)
            results.append({
                "layer": layer,
                "accuracy_score": accuracy_score(y_test, y_pred),
                "precision_score": precision_score(y_test, y_pred, average='weighted'),
                "recall_score": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            })

        df = pd.DataFrame(results)
        df.to_csv(f"results/similarity/accuracy_plots/acc_ankh_{label_type}_{len(indexes)}.csv")

        layers = list(df["layer"])

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot each metric
        plt.plot(layers, list(df["accuracy_score"]), label='Accuracy', marker='^')
        plt.plot(layers, list(df["precision_score"]), label='Precision', marker='o')
        plt.plot(layers, list(df["recall_score"]), label='Recall', marker='s')
        plt.plot(layers, list(df["f1_score"]), label='F1 Score', marker='d')

        # Adding labels, title, and grid
        plt.title('Layer-wise Metrics')
        plt.xlabel('Layer Number')
        plt.ylabel('Metric Value')
        plt.ylim(0, 1)  # Assuming metrics range from 0 to 1
        plt.xticks(layers)  # Show all layer numbers on the x-axis
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot instead of showing it
        output_path = f"results/similarity/accuracy_plots/accuracies_ankh_{label_type}.png"  # Specify the file name
        plt.savefig(output_path)  # Save with high resolution
        plt.close()  # Close the plot to free memory
    del ankh_embs

    protgpt2_embs = load_data("protgpt2_merged_tensor.pt")

    for label_type in tqdm(common_labels.keys()):
        indexes = filter_by_multiple_labels(all_labels, label_type, label_values=common_labels[label_type])
        results = []
        embs = protgpt2_embs[indexes]
        labels = list(all_labels[label_type][indexes])
        for layer in tqdm(range(embs.shape[1])):
            layer_embs = embs[:, layer, :]
            transformed = calculate_lda(layer_embs, labels, "protgpt2", label_type, layer)

            X_train, X_test, y_train, y_test = train_test_split(transformed, labels, test_size=0.2, random_state=42)

            # Train KNN classifier
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)

            # Predict on the test set
            y_pred = knn.predict(X_test)
            results.append({
                "layer": layer,
                "accuracy_score": accuracy_score(y_test, y_pred),
                "precision_score": precision_score(y_test, y_pred, average='weighted'),
                "recall_score": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            })

        df = pd.DataFrame(results)
        df.to_csv(f"results/similarity/accuracy_plots/acc_protgpt2_{label_type}_{len(indexes)}.csv")

        layers = list(df["layer"])

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot each metric
        plt.plot(layers, list(df["accuracy_score"]), label='Accuracy', marker='^')
        plt.plot(layers, list(df["precision_score"]), label='Precision', marker='o')
        plt.plot(layers, list(df["recall_score"]), label='Recall', marker='s')
        plt.plot(layers, list(df["f1_score"]), label='F1 Score', marker='d')

        # Adding labels, title, and grid
        plt.title('Layer-wise Metrics')
        plt.xlabel('Layer Number')
        plt.ylabel('Metric Value')
        plt.ylim(0, 1)  # Assuming metrics range from 0 to 1
        plt.xticks(layers)  # Show all layer numbers on the x-axis
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot instead of showing it
        output_path = f"results/similarity/accuracy_plots/accuracies_protgpt2_{label_type}.png"  # Specify the file name
        plt.savefig(output_path)  # Save with high resolution
        plt.close()  # Close the plot to free memory

    del protgpt2_embs
