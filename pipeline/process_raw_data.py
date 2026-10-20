import json
import pandas as pd
from pathlib import Path

from utils.eda import eda
from utils.helpers import *
from utils.preprocessing import encode_column_ohe, handle_missing_values
from configs.settings import (
    ENCODING_MAP, RAW_DATA_TEST, RAW_DATA_TRAIN, PROCESSED_DATA_TEST, 
    PROCESSED_DATA_TRAIN, METADATA
)

def main(csv_input_path: Path, metadata_output_path: Path, csv_output_path: Path, train_file: bool = True) -> pd.DataFrame:
    csv_input_file = Path(csv_input_path)
    metadata_dir = Path(metadata_output_path)
    csv_output_file = Path(csv_output_path)
    suffix = "train" if train_file else "test"

    if not check_file_exists(csv_input_file, "csv", create_if_missing=False):
        raise FileNotFoundError(f"Required input CSV not found: {csv_input_file}")

    df = pd.read_csv(csv_input_file)
    encoding_meta_registry = {}
    fill_missing_meta_registry = {}

    train_encoding_path = metadata_dir / "encoding_metadata_train.json"
    train_encoding_meta = None
    if not train_file:
        if not train_encoding_path.exists():
            raise FileNotFoundError(
                f"Test encoding needs train encoding metadata at: {train_encoding_path}. "
                f"Run with train_file=True first."
            )
        with open(train_encoding_path, "r", encoding="utf-8") as f:
            train_encoding_meta = json.load(f)

    for encoding_type, config in ENCODING_MAP.items():
        items = config.items() if isinstance(config, dict) else [(col, None) for col in config]
        for col, label_map in items:
            if col not in df.columns:
                continue
            if encoding_type == "ohe_encoding":
                if train_file:
                    dummies, meta = encode_column_ohe(df, column=col, drop_first=True)
                    encoding_type_label = "one_hot"
                else:
                    dummies, meta = apply_ohe_with_reference(
                        df, column=col, reference_meta=train_encoding_meta[col]["metadata"]
                    )
                    encoding_type_label = "one_hot_applied_from_train"
                    if meta["unseen_categories"]:
                        print(f"WARNING: '{col}' has categories unseen in train: {meta['unseen_categories']}")

                encoding_meta_registry[col] = {"encoding_type": encoding_type_label, "metadata": meta}
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if not bool_cols.empty:
        df[bool_cols] = df[bool_cols].astype(int)

    for col in list(df.columns):
        df, metadata = handle_missing_values(df, col)
        fill_missing_meta_registry[col] = metadata

    remaining_categorical = df.select_dtypes(include=["object", "str", "category"]).columns
    if not remaining_categorical.empty:
        print(f"Columns left as string/object dtype (not encoded): {list(remaining_categorical)}")

    if encoding_meta_registry:
        encoding_path = metadata_dir / f"encoding_metadata_{suffix}.json"
        check_file_exists(encoding_path, "json", create_if_missing=True)
        with open(encoding_path, "w", encoding="utf-8") as f:
            json.dump(encoding_meta_registry, f, indent=4, default=str)
        print(f"Encoding metadata successfully saved to: {encoding_path}")

    if fill_missing_meta_registry:
        fill_missing_path = metadata_dir / f"fill_missing_{suffix}.json"
        check_file_exists(fill_missing_path, "json", create_if_missing=True)
        with open(fill_missing_path, "w", encoding="utf-8") as f:
            json.dump(fill_missing_meta_registry, f, indent=4, default=str)
        print(f"Missing-value metadata successfully saved to: {fill_missing_path}")

    eda_output_dir = metadata_dir.parent / "eda_reports" / suffix
    eda(df, eda_output_dir)

    check_file_exists(csv_output_file, "csv", create_if_missing=True)
    df.to_csv(csv_output_file, index=False)
    print(f"Processed dataset successfully saved to: {csv_output_file.resolve()}")

    return df

if __name__ == "__main__":
    main(RAW_DATA_TRAIN, METADATA, PROCESSED_DATA_TRAIN, train_file=True)
    main(RAW_DATA_TEST, METADATA, PROCESSED_DATA_TEST, train_file=False)
