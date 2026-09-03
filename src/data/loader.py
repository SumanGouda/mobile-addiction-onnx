"""Data loading utilities."""
from pathlib import Path
from typing import Union
import pandas as pd


def load_raw_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load a raw CSV file into a DataFrame.

    Args:
        path: File path to the raw CSV file as a string or Path object.

    Returns:
        The loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the file is not a valid non-empty CSV format.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at path: {file_path.resolve()}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a valid file: {file_path.resolve()}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"The specified data file is empty: {file_path.resolve()}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file at {file_path.resolve()}: {str(e)}") from e

    return df
