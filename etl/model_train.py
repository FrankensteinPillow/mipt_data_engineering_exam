import json
import pickle
import pprint

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from pathlib import Path


def model_train(base_path: Path):
    X_train: np.array = np.load(base_path.joinpath("X_train.npy"))
    y_train: np.array = np.load(base_path.joinpath("y_train.npy"))
    X_test: np.array = np.load(base_path.joinpath("X_test.npy"))
    y_test: np.array = np.load(base_path.joinpath("y_test.npy"))

    model = LogisticRegression(random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
    }
    pprint.pprint(results)
    # сохраним результаты предсказаний
    np.savetxt(base_path.joinpath("y_pred.csv"), y_pred, delimiter=",")
    # сохраним результаты метрик полученных на тестовой выборке
    with open(base_path.joinpath("metrics.json"), "w") as f:
        json.dump(results, f)
    # сохраним саму модель как pickle файл
    with open(base_path.joinpath("logistic_regression.pickle"), "wb") as f:
        pickle.dump(model, f)
