import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

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
X_scaled = StandardScaler().fit_transform(X)

# K-Means
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

k_labels = kmeans.fit_predict(X_scaled)

k_score = silhouette_score(
    X_scaled,
    k_labels
)

# Hierarchical
hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

h_labels = hierarchical.fit_predict(X_scaled)

h_score = silhouette_score(
    X_scaled,
    h_labels
)

print("K-Means Silhouette Score:")
print(k_score)

print("\nHierarchical Silhouette Score:")
print(h_score)

print("\nConclusion:")

if k_score > h_score:
    print("K-Means performed better.")
elif h_score > k_score:
    print("Hierarchical Clustering performed better.")
else:
    print("Both methods performed similarly.")