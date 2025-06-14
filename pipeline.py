import datetime as dt
from pathlib import Path
from uuid import uuid4

from etl.data_load import load_data
from etl.show_statistics import show_statistics
from etl.data_preprocessing import data_preprocessing
from etl.model_train import model_train


def main():
    cur_dt: str = dt.datetime.strftime(dt.datetime.now(), "%H_%M_%d_%m_%Y")
    id_part = str(uuid4())[:8]
    result_dir = (
        Path(__file__)
        .parent.resolve()
        .joinpath(f"results/task_{id_part}_{cur_dt}")
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1.
    load_data(result_dir)
    # 2.
    show_statistics(result_dir)
    # 3.
    data_preprocessing(result_dir)
    # 4.
    model_train(result_dir)


if __name__ == "__main__":
    main()
