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

print("First Five Records:")
print(df.head())

print("\nLast Five Records:")
print(df.tail())