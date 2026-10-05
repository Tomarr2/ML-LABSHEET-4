import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

df = pd.DataFrame(X, columns=features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

print("Standardized Dataset:")
print(X_scaled[:5])