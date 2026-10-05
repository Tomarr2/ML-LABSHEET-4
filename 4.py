import pandas as pd
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

df = pd.DataFrame(X, columns=features)

# Select numerical features
X_features = df[
    ["sepal_length", "sepal_width", "petal_length", "petal_width"]
]

print("Selected Features:")
print(X_features.head())