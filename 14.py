import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

# Before scaling
model1 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels1 = model1.fit_predict(X)

# After scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model2 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels2 = model2.fit_predict(X_scaled)

plt.scatter(X[:, 0], X[:, 1], c=labels1)
plt.title("Before Scaling")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels2)
plt.title("After Scaling")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()