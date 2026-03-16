"""Utilities for loading raw datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".csv", ".parquet"}
DEFAULT_RAW_DATA_DIR = Path("data/raw")


def configure_logging() -> None:
    """Configure a simple logger if the application has not done it yet."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def resolve_data_path(filename: str | None = None, data_dir: Path = DEFAULT_RAW_DATA_DIR) -> Path:
    """Resolve dataset path inside the raw data directory."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory was not found: '{data_dir}'. Create it and add a CSV or Parquet file."
        )

    if filename:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file was not found: '{path}'. Put the file into '{data_dir}'."
            )
        return path

    supported_files = sorted(
        path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not supported_files:
        raise FileNotFoundError(
            f"No supported dataset files were found in '{data_dir}'. Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}."
        )
    return supported_files[0]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from CSV or Parquet and log basic metadata."""
    configure_logging()

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: '{dataset_path}'. Check the file name and location."
        )

    suffix = dataset_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{suffix}' for '{dataset_path.name}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}."
        )

    if suffix == ".csv":
        dataframe = pd.read_csv(dataset_path)
    else:
        dataframe = pd.read_parquet(dataset_path)

    LOGGER.info("Loaded dataset from %s", dataset_path)
    LOGGER.info("Dataset shape: %s", dataframe.shape)
    LOGGER.info("Columns: %s", list(dataframe.columns))
    LOGGER.info("Dtypes: %s", {column: str(dtype) for column, dtype in dataframe.dtypes.items()})
    return dataframe
