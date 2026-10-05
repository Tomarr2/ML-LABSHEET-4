import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

X_scaled = StandardScaler().fit_transform(X)

# Before PCA
model1 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels1 = model1.fit_predict(X_scaled)

score1 = silhouette_score(X_scaled, labels1)

# After PCA
X_pca = PCA(n_components=2).fit_transform(X_scaled)

model2 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels2 = model2.fit_predict(X_pca)

score2 = silhouette_score(X_pca, labels2)

print("Silhouette Score Before PCA:", score1)
print("Silhouette Score After PCA:", score2)