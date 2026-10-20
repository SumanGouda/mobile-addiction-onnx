import pandas as pd

def handle_missing_values(df: pd.DataFrame, col: str,) -> tuple[pd.DataFrame, dict]:
    """
    Handles missing values and returns metadata describing
    the operation performed.
    """
    metadata = {
        "column": col,
        "missing_count": None,
        "missing_percentage": None,
        "action": None,
        "fill_strategy": None,
        "fill_value": None,
        "indicator_column": None,
    }
    if col not in df.columns:
        metadata["action"] = "column_not_found"
        print(f"Column '{col}' not found in DataFrame.")
        return df, metadata
    
    missing_count = df[col].isnull().sum()
    total_rows = len(df)
    metadata["missing_count"] = int(missing_count)

    if total_rows == 0:
        metadata["action"] = "empty_dataframe"
        print("DataFrame is empty.")
        return df, metadata

    missing_pct = (missing_count / total_rows) * 100
    metadata["missing_percentage"] = round(missing_pct, 2)
    print(f"{col}: {missing_pct:.2f}% missing", end=" → ")

    # > 50% missing → drop column
    if missing_pct > 50:
        df = df.drop(columns=[col])
        metadata["action"] = "drop_column"
        print("Dropped column (>50% missing)")

    # 30–50% missing → drop rows
    elif missing_pct > 30:
        rows_before = len(df)
        df = df.dropna(subset=[col]).reset_index(drop=True)
        metadata["action"] = "drop_rows"
        metadata["rows_dropped"] = rows_before - len(df)
        print("Dropped rows (30-50% missing)")

    # 10–30% missing → fill + indicator
    elif missing_pct > 10:
        indicator_col = f"{col}_was_missing"
        df[indicator_col] = df[col].isnull().astype(int)

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].mean()
            metadata["fill_strategy"] = "mean"
        else:
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
            metadata["fill_strategy"] = "mode"

        df[col] = df[col].fillna(fill_val)
        metadata["action"] = "fill_and_indicator"
        metadata["fill_value"] = (
            fill_val.item() if hasattr(fill_val, "item") else fill_val
        )
        metadata["indicator_column"] = indicator_col
        print(f"Filled + added indicator column '{indicator_col}'")

    # < 10% missing → fill only
    elif missing_pct > 0:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            metadata["fill_strategy"] = "median"
        else:
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
            metadata["fill_strategy"] = "mode"

        df[col] = df[col].fillna(fill_val)
        metadata["action"] = "fill"
        metadata["fill_value"] = (
            fill_val.item() if hasattr(fill_val, "item") else fill_val
        )
        print(f"Filled with {metadata['fill_strategy']}")
    
    else:
        metadata["action"] = "no_missing_values"
        print("No missing values")

    return df, metadata

def encode_column_ohe(df: pd.DataFrame, column: str, drop_first: bool = True) -> tuple[pd.DataFrame, dict]:
    """Generates one-hot encoded dummies and tracks category mappings and dropped reference level."""
    categories = sorted(df[column].dropna().unique().tolist())
    dummies = pd.get_dummies(df[column], prefix=column, drop_first=drop_first, dtype=int)
    dropped_feature = categories[0] if (drop_first and categories) else None
    metadata = {
        "original_column"           : column,
        "all_categories"            : categories,
        "encoded_columns"           : list(dummies.columns),
        "dropped_baseline_category" : dropped_feature,
    }
    return dummies, metadata
