import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def plot_numeric_vs_target(df: pd.DataFrame, numeric_cols: list, target_col: str, output_dir: str = None):
    """
    For numeric features vs. binary target:
    - Boxplots of feature by target class.
    - Violin plots.
    """
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=target_col, y=col, data=df)
        plt.title(f"{col} by {target_col}")
        plt.tight_layout()
        if output_dir:
            plt.savefig(f"{output_dir}/bivariate_num_{col}_vs_{target_col}.png")
        plt.show()

def plot_categorical_vs_target(df: pd.DataFrame, categorical_cols: list, target_col: str, output_dir: str = None):
    """
    For categorical features vs. binary target:
    - Bar plot of proportion of target=1 by category.
    """
    for col in categorical_cols:
        # Compute proportion of target=1 per category
        prop_df = df.groupby(col)[target_col].mean().reset_index()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=col, y=target_col, data=prop_df)
        plt.xticks(rotation=45)
        plt.title(f"Mean {target_col} by {col}")
        plt.tight_layout()
        if output_dir:
            plt.savefig(f"{output_dir}/bivariate_cat_{col}_vs_{target_col}.png")
        plt.show()
