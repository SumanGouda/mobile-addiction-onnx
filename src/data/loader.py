"""Data loading utilities."""
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load a raw CSV file into a DataFrame.

    Args:
        path: Path to the raw CSV file.

    Returns:
        The loaded DataFrame, unmodified.
    """
    return pd.read_csv(path)