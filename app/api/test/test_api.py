"""Helper to sample a random raw row from test.csv and send it to the /predict API."""
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # project root
TEST_CSV_PATH = str(PROJECT_ROOT / "data" / "raw" / "test.csv")
API_URL = "http://127.0.0.1:8000/predict"


def get_data_dict(csv_path: str = None) -> Dict:
    """Pick a random complete (no missing values) row from the raw test
    CSV and return it as a dict shaped for the /predict API — i.e. with
    the `id` column dropped, since the model doesn't use it.

    Args:
        csv_path: Path to the raw CSV to sample from. Defaults to
            data/raw/test.csv.

    Returns:
        A dict of raw feature values, ready to send as a JSON request body.
    """
    path = Path(csv_path) if csv_path else TEST_CSV_PATH
    df = pd.read_csv(path)

    df = df.drop(columns=["id"], errors="ignore")

    # Only sample rows with no missing values — the API requires every
    # field to be present.
    complete_rows = df.dropna()
    if complete_rows.empty:
        raise ValueError("No complete (non-missing) rows found to sample from.")

    row = complete_rows.sample(n=1).iloc[0]
    return row.to_dict()


if __name__ == "__main__": 
    payload = get_data_dict(TEST_CSV_PATH)

    # Automated E2E Test
    try:
        response = requests.post(API_URL, json=payload)
         
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(response.json())
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {API_URL}. Is your FastAPI server running?")
