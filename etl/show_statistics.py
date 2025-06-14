import pandas as pd
from pathlib import Path


def show_statistics(base_path: str):
    base_path = Path(base_path)
    data: pd.DataFrame = pd.read_csv(
        base_path.joinpath("breast_cancer.csv"), sep=","
    )
    print(data.info())
    print(data.describe())
