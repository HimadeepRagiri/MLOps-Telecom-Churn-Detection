import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarizing missing values per column.
    """
    missing = df.isnull().sum()
    pct = missing / len(df) * 100
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return summary

def plot_missing_heatmap(df: pd.DataFrame, output_dir: str = None):
    """
    Plot a heatmap indicating missing data locations.
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Data Heatmap")
    plt.tight_layout()
    if output_dir:
        plt.savefig(f"{output_dir}/missing_heatmap.png")
    plt.show()
