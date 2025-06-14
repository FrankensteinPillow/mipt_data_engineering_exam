from sklearn.datasets import load_breast_cancer
from pathlib import Path
from logging import getLogger


logger = getLogger(__name__)


def load_data(base_path: str) -> None:
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer(return_X_y=False, as_frame=True)
    dataset_file_path = base_path.joinpath(data["filename"])
    data["frame"].to_csv(dataset_file_path)
    logger.info(f"Dataset saved to file: {dataset_file_path}")
