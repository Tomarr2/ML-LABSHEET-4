import pandas as pd
from sklearn.datasets import load_iris

# Load Iris data directly
X, y = load_iris(return_X_y=True)

# Feature names
features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

# Create DataFrame
df = pd.DataFrame(X, columns=features)

# Add target
df["target"] = y

print("Dataset loaded successfully!")
print(df)