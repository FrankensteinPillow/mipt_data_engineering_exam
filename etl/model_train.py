import json
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)


def model_train(base_path: str):
    base_path = Path(base_path)
    X_train: np.array = np.load(base_path.joinpath("X_train.npy"))
    y_train: np.array = np.load(base_path.joinpath("y_train.npy"))
    X_test: np.array = np.load(base_path.joinpath("X_test.npy"))
    y_test: np.array = np.load(base_path.joinpath("y_test.npy"))

    model = LogisticRegression(random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
    }
    logger.info(f"metrics: {metrics}")

    np.save(base_path.joinpath("y_pred.npy"), y_pred)
    logger.info(f"prediction saved to {base_path.joinpath('y_pred.npy')}")

    with open(base_path.joinpath("metrics.json"), "w") as f:
        json.dump(metrics, f)
    logger.info(f"metrics saved to {base_path.joinpath('metrics.json')}")

    with open(base_path.joinpath("logistic_regression.pickle"), "wb") as f:
        pickle.dump(model, f)
    logger.info(
        f"model saved to {base_path.joinpath('logistic_regression.pickle')}"
    )
