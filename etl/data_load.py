from sklearn.datasets import load_breast_cancer
from pathlib import Path


def load_data(base_path: Path) -> None:
    data = load_breast_cancer(return_X_y=False, as_frame=True)
    dataset_file_path = base_path.joinpath(data["filename"])
    data["frame"].to_csv(dataset_file_path)
    print(f"Dataset file path: {dataset_file_path}")
