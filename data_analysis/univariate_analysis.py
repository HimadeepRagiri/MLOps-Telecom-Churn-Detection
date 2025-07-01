import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: list, output_dir: str = None):
    """
    For each numeric column, plot histogram and boxplot side by side.
    - numeric_cols: list of column names of numeric features.
    - output_dir: if provided, saves plots as PNG.
    """
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f"Histogram of {col}")
        sns.boxplot(x=df[col].dropna(), ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}")
        plt.tight_layout()
        if output_dir:
            plt.savefig(f"{output_dir}/univariate_{col}.png")
        plt.show()

def plot_categorical_counts(df: pd.DataFrame, categorical_cols: list, output_dir: str = None):
    """
    For each categorical column, plot bar plot of value counts.
    """
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=vc.index.astype(str), y=vc.values)
        plt.xticks(rotation=45)
        plt.title(f"Value counts of {col}")
        plt.tight_layout()
        if output_dir:
            plt.savefig(f"{output_dir}/univariate_cat_{col}.png")
        plt.show()
