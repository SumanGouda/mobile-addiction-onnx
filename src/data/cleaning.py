"""Missing value imputation utilities."""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fill_missing_by_correlation(
    df: pd.DataFrame,
    feature: str,
    threshold: float = 0.3,
    missing_threshold: float = 0.6,
    n_related: int = 2,
) -> Tuple[pd.DataFrame, str]:
    """Fill missing values in a numeric column using regression on its most
    correlated features.

    Args:
        df: The DataFrame containing the feature to fill.
        feature: Name of the column to impute.
        threshold: Minimum absolute correlation required for a column to be
            used as a predictor.
        missing_threshold: If the fraction of missing values in `feature`
            exceeds this, filling is skipped entirely.
        n_related: Maximum number of correlated predictor columns to use.

    Returns:
        A tuple of (possibly modified DataFrame, a message describing what
        happened).
    """
    if df[feature].dtype == "object":
        return df, f"'{feature}' is an object/string column — skipped."

    missing_pct = df[feature].isna().mean()
    if missing_pct > missing_threshold:
        return df, (
            f"'{feature}' has {missing_pct:.0%} missing values "
            f"(> {missing_threshold:.0%}) — skipped, too much data missing "
            "to reliably fill."
        )

    if missing_pct == 0:
        return df, f"'{feature}' has no missing values — nothing to do."

    numeric_df = df.select_dtypes(include=np.number)
    if feature not in numeric_df.columns:
        return df, f"'{feature}' is not numeric — skipped."

    corr = numeric_df.corr()[feature].drop(feature).abs().sort_values(ascending=False)
    top_corr = corr[corr > threshold].head(n_related)

    if top_corr.empty:
        return df, (
            f"No feature correlates with '{feature}' above threshold "
            f"{threshold} — treating as independent, skipped."
        )

    predictors = top_corr.index.tolist()

    valid_rows = df[predictors].notna().all(axis=1)
    train_mask = valid_rows & df[feature].notna()
    predict_mask = valid_rows & df[feature].isna()

    if predict_mask.sum() == 0:
        return df, (
            f"Rows missing '{feature}' also have missing predictor values "
            f"({predictors}) — can't fill via regression, skipped."
        )

    model = LinearRegression()
    model.fit(df.loc[train_mask, predictors], df.loc[train_mask, feature])

    predicted = model.predict(df.loc[predict_mask, predictors])
    df.loc[predict_mask, feature] = predicted

    still_missing = df[feature].isna().sum()
    msg = (
        f"Filled {predict_mask.sum()} missing values in '{feature}' using "
        f"{predictors} (LinearRegression)."
    )
    if still_missing > 0:
        msg += f" {still_missing} rows still missing (predictors were also null)."

    return df, msg


def fill_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback fill: fill any remaining missing numeric values with the
    column median.

    Args:
        df: The DataFrame to fill.

    Returns:
        The DataFrame with remaining numeric missing values filled.
    """
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
    return df
