import numpy as np
from sklearn.neighbors import NearestNeighbors


def calculate_mu(data, nn=3):
    knn = NearestNeighbors(n_neighbors=nn).fit(data)
    distances, _ = knn.kneighbors(data)

    # The first column is the distance to the point itself (0), skip it
    r1 = distances[:, 1]  # First nearest neighbor
    r2 = distances[:, 2]

    # Step 2: Compute the ratio μ_i = r2 / r1
    mu = r2 / r1

    return np.linalg.norm(mu)
