import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_dataset():
    data = fetch_california_housing()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["TargetPrice"] = data.target

    return df
