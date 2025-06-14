from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_preprocessing(base_path: str):
    base_path = Path(base_path)
    data: pd.DataFrame = pd.read_csv(
        base_path.joinpath("breast_cancer.csv"), sep=","
    )
    X, y = (
        data[[c for c in data.columns if c != "target"]],
        data["target"],
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )
    np.save(base_path.joinpath("X_scaled.npy"), X_scaled)
    np.save(base_path.joinpath("X_train.npy"), X_train)
    np.save(base_path.joinpath("X_test.npy"), X_test)
    np.save(base_path.joinpath("y_train.npy"), y_train)
    np.save(base_path.joinpath("y_test.npy"), y_test)
