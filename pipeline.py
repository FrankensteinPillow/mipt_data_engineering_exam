import datetime as dt
import logging.config
from pathlib import Path
from uuid import uuid4
import logging

from etl.data_load import load_data
from etl.show_statistics import show_statistics
from etl.data_preprocessing import data_preprocessing
from etl.model_train import model_train


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
        "simple": {
            "format": "%(levelname)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "pipeline.log",
            "mode": "a",
            "encoding": "utf-8",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file_handler"],
    },
    "loggers": {
        "pipeline": {
            "level": "DEBUG",
            "handlers": ["console", "file_handler"],
            "propagate": False,
        }
    },
}


def main():
    cur_dt: str = dt.datetime.strftime(dt.datetime.now(), "%H_%M_%d_%m_%Y")
    id_part = str(uuid4())[:8]
    result_dir = (
        Path(__file__)
        .parent.resolve()
        .joinpath(f"results/task_{id_part}_{cur_dt}")
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(__file__).parent.resolve() / f"logs/task_{id_part}_{cur_dt}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path: Path = log_dir / "pipeline.log"
    LOGGING_CONFIG["handlers"]["file_handler"]["filename"] = log_file_path
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger("pipeline")

    logger.info("Start pipeline")
    logger.info(f"log file path: {log_file_path}")
    logger.info(f"result directory: {result_dir}")

    # 1.
    load_data(str(result_dir))
    # 2.
    show_statistics(str(result_dir))
    # 3.
    data_preprocessing(str(result_dir))
    # 4.
    model_train(str(result_dir))

    logger.info("Pipeline end")


if __name__ == "__main__":
    main()
