import os

import numpy as np
import pandas as pd


def get_correlations(folder):


    # Assuming your array and DataFrame are defined as follows:
    folder_path = f"{folder}/similarity/accuracy_plots"

    # The array to correlate with
    array = np.load(f"{folder}/int_dim_layers_ankh.npy")

    target_columns = ["accuracy_score", "precision_score", "recall_score", "f1_score"]

    # Dictionary to store correlations for each file
    correlation_results = {}

    # Iterate through all CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and "ankh" in filename:
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            df = df[target_columns]

            # Calculate correlations
            correlations = df.apply(lambda col: np.corrcoef(array, col)[0, 1], axis=0)

            # Store the results in the dictionary
            correlation_results[filename] = correlations


    pd.DataFrame(correlation_results).to_csv(f"{folder}/correlation.csv")

    # Print the correlation results
    for file, correlations in correlation_results.items():
        print(f"Correlations for {file}:\n{correlations}\n")
