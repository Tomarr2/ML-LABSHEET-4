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

print("Dataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nData Types:")
print(df.dtypes)