# from calculations.intrinsic_dimension_different_classes import calculate_fold_mean, calculate_sf_mean, calculate_f_mean, calculate_class_mean, calculate_type_mean
#
# res = calculate_fold_mean()
# print(max(res[1]))
# print(min(res[1]))
# print(max(res[2]))
# print(min(res[2]))
# # print(res)
# res = calculate_sf_mean()
# print(max(res[1]))
# print(min(res[1]))
# print(max(res[2]))
# print(min(res[2]))
# # print(res)
# res = calculate_f_mean()
# print(max(res[1]))
# print(min(res[1]))
# print(max(res[2]))
# print(min(res[2]))
# # print(res)
# res = calculate_class_mean()
# print(max(res[1]))
# print(min(res[1]))
# print(max(res[2]))
# print(min(res[2]))
# # print(res)
# res = calculate_type_mean()
# print(max(res[1]))
# print(min(res[1]))
# print(max(res[2]))
# print(min(res[2]))
# # print(res)


# from visualizations.sequence_lengths import run_all
#
# run_all()



# from calculations.intrinsic_dimension_data import run_all
#
# run_all()

# from visualizations.intrinsic_dimension_data import run_all
#
# run_all()




# from calculations.intrinsic_dimension_layers import run_all
#
# run_all()


# ID Through Layers
# from visualizations.intrinsic_dimension_layers import plot_all
#
# plot_all()

# from visualizations.intrinsic_dimension_layers import plot_all_filtered
#
# plot_all_filtered()


# Grandmother cells for different models/layers/labels
# from visualizations.grandmother_cells_all import run_all
#
# run_all()


# Get similarity plots and accuracies for different models/layers/labels
# from calculations.similarity import run_all
#
# run_all()



# from visualizations.tsne import run_all
#
# run_all()


# from visualizations.similarity import run_all_w_real_distance
#
# run_all_w_real_distance()



# from calculations.pca import run_all
#
# run_all()
#
#
# from visualizations.distance_histogram import run_all
#
# run_all()


# from visualizations.pca import run_all
#
# run_all()


# from visualizations.pca_colored import run_all
#
# run_all()

# from visualizations.distance_hist_emb_color import run_all
#
# run_all()

# from visualizations.scree_plot import run_all
#
# run_all()


# from visualizations.lda_id_correlation import get_correlations
#
# get_correlations("results")


# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# def compute_class_scatter_matrices(X, y):
#     """
#     Computes the between-class scatter matrix (S_B) and
#     the within-class scatter matrix (S_W) for dataset X, y.
#     """
#     # Number of features
#     n_features = X.shape[1]
#     # Overall mean of the entire dataset
#     mean_overall = np.mean(X, axis=0)
#     # Initialize S_W and S_B
#     S_W = np.zeros((n_features, n_features))
#     S_B = np.zeros((n_features, n_features))
#     # Get unique class labels
#     classes = np.unique(y)
#     for c in classes:
#         # Extract samples of class c
#         X_c = X[y == c]
#         # Class mean
#         mean_c = np.mean(X_c, axis=0)
#         # Number of samples in class c
#         N_c = X_c.shape[0]
#         # Within-class scatter contribution
#         # (X_c - mean_c) has shape [N_c, n_features]
#         # We need to sum up (x - mean_c)(x - mean_c)^T for each sample
#         # A vectorized way to compute this is:
#         X_c_centered = X_c - mean_c
#         S_W_c = X_c_centered.T.dot(X_c_centered)
#         S_W += S_W_c
#         # Between-class scatter contribution
#         mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
#         S_B_c = N_c * (mean_diff).dot(mean_diff.T)
#         S_B += S_B_c
#     return S_B, S_W
#
#
# def class_separation_score(X, y):
#     """
#     Computes a scalar class separation score as Trace(S_B) / Trace(S_W).
#     """
#     S_B, S_W = compute_class_scatter_matrices(X, y)
#     # To avoid division by zero in degenerate cases
#     trace_sw = np.trace(S_W)
#     if trace_sw == 0:
#         return np.inf  # or return 0, or handle as you see fit
#     return np.trace(S_B) / trace_sw
#
#
# data = load_iris()
# X = data.data
# y = data.target
#
# # 2. Fit an LDA model (optional for demonstration)
# lda = LinearDiscriminantAnalysis()
# lda.fit(X, y)
#
# # 3. Compute class separation score
# score = class_separation_score(X, y)
# print("Class separation score (Trace(S_B)/Trace(S_W)):", score)
