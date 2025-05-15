
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_preprocess():
    data = load_iris()
    X, y = data.data, data.target

    # Simulate missing values
    X[0][0] = np.nan
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
