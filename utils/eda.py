from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math


def eda(df: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Orchestrates all EDA visualization functions and saves dark-themed plots to disk.

    Returns:
        dict: Mapping of plot types to their saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define distinct destination paths for each plot
    generated_plots = {
        "correlation": output_dir / "correlation_heatmap.jpg",
        "outliers": output_dir / "numerical_outliers.jpg",
        "distributions": output_dir / "numerical_distributions.jpg",
        "categorical_counts": output_dir / "categorical_counts.jpg",
    } 
    _plot_correlation_heatmap(df, generated_plots["correlation"])
    _plot_numerical_outliers(df, generated_plots["outliers"])
    _plot_numerical_distributions(df, generated_plots["distributions"])
    _plot_categorical_counts(df, generated_plots["categorical_counts"])

    return
    

def _plot_numerical_distributions(df: pd.DataFrame, output_path: Path, ncols: int = 3) -> Path:
    """Generates a dark-themed multi-grid canvas showing PDF (KDE) curves and histograms for numerical features."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Automatically detect numerical features
    num_df = df.select_dtypes(include=[np.number])
    num_cols = num_df.columns.tolist()

    if not num_cols:
        print("Skipping distribution plot: No numerical columns found.")
        return output_path

    # 2. Compute dynamic layout bounds
    total_features = len(num_cols)
    nrows = math.ceil(total_features / ncols)

    fig_width = ncols * 4.5
    fig_height = nrows * 4.0

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        facecolor="#0E1117",
    )

    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # 3. Plot histogram + KDE (PDF curve) for each feature
    for idx, col in enumerate(num_cols):
        ax = axes_list[idx]
        ax.set_facecolor("#0E1117")

        sns.histplot(
            num_df[col].dropna(),
            kde=True,
            stat="density",
            ax=ax,
            color="#1F77B4",
            edgecolor="#0E1117",
            line_kws={"color": "#00FFC8", "linewidth": 2},
        )

        ax.set_title(f"{col} (PDF)", color="white", fontsize=12, pad=10)
        ax.set_xlabel("", color="white")
        ax.set_ylabel("Density", color="white", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", colors="white", labelsize=9)

    # 4. Hide unused grid subplots
    for idx in range(total_features, len(axes_list)):
        fig.delaxes(axes_list[idx])

    fig.suptitle("Numerical Probability Density Functions (KDE)", color="white", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return output_path

def _plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> Path:
    """Generates a dark-themed correlation heatmap for numerical columns and saves it as an image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter to numerical columns only
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or num_df.shape[1] < 2:
        print("Skipping correlation plot: Need at least 2 numerical columns.")
        return output_path

    corr_matrix = num_df.corr()  
    fig_width = max(8, num_df.shape[1] * 1.2)
    fig_height = max(6, num_df.shape[1] * 1.0) 
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        linecolor="#1E222A",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", color="white", fontsize=14, pad=15)
    plt.xticks(color="white", rotation=45, ha="right")
    plt.yticks(color="white", rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return output_path

def _plot_categorical_counts(df: pd.DataFrame, output_path: Path) -> Path:
    """Generates a dark-themed multi-grid chart for all categorical features in df."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    cat_cols = [
        col for col in df.columns 
        if df[col].dtype == "object" or df[col].dtype.name == "category"
    ] 
    if not cat_cols:
        print("Skipping categorical plot: No categorical columns found.")
        return output_path 
    top_n = 10
    ncols = 2
    num_cols = len(cat_cols)
    nrows = math.ceil(num_cols / ncols) 
    fig_width = ncols * 8.0
    fig_height = nrows * 4.5

    plt.style.use("dark_background")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), facecolor="#0E1117",) 
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, col in enumerate(cat_cols):
        ax = axes_list[idx]
        ax.set_facecolor("#0E1117")

        counts = df[col].value_counts().head(top_n)
 
        sns.barplot(x=counts.values, y=counts.index, hue=counts.index, palette="Blues_r", legend=False, ax=ax,) 
        # Annotate values
        max_val = counts.max() if not counts.empty else 1
        for i, val in enumerate(counts.values):
            ax.text(
                val + (max_val * 0.01),
                i,
                f" {val:,}",
                va="center",
                color="white",
                fontsize=9,
            ) 
        ax.set_title(f"{col} (Top {top_n})", color="white", fontsize=12, pad=10)
        ax.set_xlabel("Count", color="white", fontsize=10)
        ax.set_ylabel("", color="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", colors="white", labelsize=9)
 
    for idx in range(num_cols, len(axes_list)):
        fig.delaxes(axes_list[idx]) 
    fig.suptitle("Categorical Feature Distributions", color="white", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig) 
    return output_path

def _plot_numerical_outliers(df: pd.DataFrame, output_path: Path, ncols: int = 3) -> Path:
    """Generates a dark-themed multi-grid canvas of box plots to detect outliers across all numerical columns."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    num_df = df.select_dtypes(include=[np.number])
    num_cols = num_df.columns.tolist()

    if not num_cols:
        print("Skipping outlier plot: No numerical columns found.")
        return output_path 
    total_features = len(num_cols)
    nrows = math.ceil(total_features / ncols) 
    fig_width = ncols * 4.5
    fig_height = nrows * 4.0 
    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        facecolor="#0E1117",
    ) 
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
 
    for idx, col in enumerate(num_cols):
        ax = axes_list[idx]
        ax.set_facecolor("#0E1117")

        sns.boxplot(
            y=num_df[col].dropna(),
            ax=ax,
            color="#1F77B4",
            flierprops={
                "marker": "o",
                "markerfacecolor": "#FF4B4B",
                "markeredgecolor": "#FF4B4B",
                "markersize": 5,
            },
            boxprops={"edgecolor": "white"},
            whiskerprops={"color": "white"},
            capprops={"color": "white"},
            medianprops={"color": "#00FFC8", "linewidth": 2},
        )

        ax.set_title(f"{col}", color="white", fontsize=12, pad=10)
        ax.set_ylabel("", color="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", colors="white", labelsize=9) 
    for idx in range(total_features, len(axes_list)):
        fig.delaxes(axes_list[idx]) 
    fig.suptitle("Numerical Outlier Analysis (1.5x IQR Rule)", color="white", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return output_path
