import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

def fill_missing_by_correlation(df, feature, threshold=0.3, missing_threshold=0.6, n_related=2):
    """
    Fill missing values in `feature` using regression on its top-N correlated
    numeric features. Returns (df, message) — df is modified only if filling happens.
    """
    if df[feature].dtype == "object":
        return df, f"'{feature}' is an object/string column — skipped."

    missing_pct = df[feature].isna().mean()
    if missing_pct > missing_threshold:
        return df, f"'{feature}' has {missing_pct:.0%} missing values (> {missing_threshold:.0%}) — skipped, too much data missing to reliably fill."

    if missing_pct == 0:
        return df, f"'{feature}' has no missing values — nothing to do."

    numeric_df = df.select_dtypes(include=np.number)
    if feature not in numeric_df.columns:
        return df, f"'{feature}' is not numeric — skipped."

    corr = numeric_df.corr()[feature].drop(feature).abs().sort_values(ascending=False)
    top_corr = corr[corr > threshold].head(n_related)

    if top_corr.empty:
        return df, f"No feature correlates with '{feature}' above threshold {threshold} — treating as independent, skipped."

    predictors = top_corr.index.tolist()

    valid_rows = df[predictors].notna().all(axis=1)
    train_mask = valid_rows & df[feature].notna()
    predict_mask = valid_rows & df[feature].isna()

    if predict_mask.sum() == 0:
        return df, f"Rows missing '{feature}' also have missing predictor values ({predictors}) — can't fill via regression, skipped."

    model = LinearRegression()
    model.fit(df.loc[train_mask, predictors], df.loc[train_mask, feature])

    predicted = model.predict(df.loc[predict_mask, predictors])
    df.loc[predict_mask, feature] = predicted

    still_missing = df[feature].isna().sum()
    msg = f"Filled {predict_mask.sum()} missing values in '{feature}' using {predictors} (LinearRegression)."
    if still_missing > 0:
        msg += f" {still_missing} rows still missing (predictors were also null)."

    return df, msg

def plot_correlation_matrix(df, figsize=(10, 8), cmap="coolwarm", annot=True):
    """Plot a dark-themed correlation matrix heatmap using numeric columns only."""
    numeric_df = df.select_dtypes(include=np.number)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize)

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle (redundant)

    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        linecolor="#2a2a2a",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Correlation Matrix", fontsize=14, pad=15, color="white")
    plt.tight_layout()
    plt.show()

    return fig

