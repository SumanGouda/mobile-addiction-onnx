"""Helper to sample a random raw row from test.csv and send it to the /predict API."""
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # project root
TEST_CSV_PATH = str(PROJECT_ROOT / "data" / "raw" / "test.csv")
API_URL = "http://127.0.0.1:8000/predict"


def get_data_dict(csv_path: str = None) -> Dict:
    path = Path(csv_path) if csv_path else TEST_CSV_PATH
    df = pd.read_csv(path)

    df = df.drop(columns=["id"], errors="ignore")

    complete_rows = df.dropna()
    if complete_rows.empty:
        raise ValueError("No complete rows found to sample from.")

    row = complete_rows.sample(n=1).iloc[0]

    # Convert NumPy data types (int64, float64) to native Python types
    return {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in row.to_dict().items()
    }


if __name__ == "__main__":
    payload = get_data_dict(TEST_CSV_PATH)

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            data = response.json()
            prob = data.get("addicted_probability", 0)
            label = "Addicted" if data.get("addicted_label") == 1 else "Not Addicted"

            print("\n" + "=" * 40)
            print("         INFERENCE API TEST RESULT       ")
            print("=" * 40)
            print(f" Status        : {response.status_code} OK")
            print(f" Prediction    : {label}")
            print(f" Probability   : {prob:.4f} ({prob * 100:.2f}%)")
            print(f" Model Run ID  : {data.get('model_run_id')}")
            print("=" * 40 + "\n")
        else:
            print(f"\n[Error {response.status_code}] {response.text}\n")

    except requests.exceptions.ConnectionError:
        print(f"\n[Connection Failure] Unable to reach {API_URL}. Is Uvicorn running?\n")