import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Dataset
X = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [5.0, 3.4, 1.5, 0.2],
    [6.4, 3.2, 4.5, 1.5],
    [6.0, 3.0, 4.8, 1.8],
    [6.3, 3.3, 4.7, 1.6],
    [6.7, 3.1, 5.6, 2.4],
    [7.0, 3.2, 5.8, 1.6],
    [6.5, 3.0, 5.5, 1.8]
])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means with K = 2
model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(X_scaled)

print("Cluster Labels for K = 2:")
print(labels)

print("\nCluster Centers:")
print(model.cluster_centers_)