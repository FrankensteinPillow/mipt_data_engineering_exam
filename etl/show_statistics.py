import pandas as pd
from pathlib import Path
from logging import getLogger
import io


logger = getLogger(__name__)


def show_statistics(base_path: str):
    base_path = Path(base_path)
    data: pd.DataFrame = pd.read_csv(
        base_path.joinpath("breast_cancer.csv"), sep=","
    )
    buf = io.StringIO()
    data.info(buf=buf)
    logger.info(buf.getvalue())
    logger.info(data.describe())
