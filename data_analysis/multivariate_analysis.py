import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as ss
import os

def plot_correlation_matrix(df: pd.DataFrame, numeric_cols: list, output_dir: str = None):
    """
    Plot heatmap of correlation matrix among numeric features.
    """
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    if output_dir:
        plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.show()

def plot_pairplot_sample(df: pd.DataFrame, numeric_cols: list, sample_frac: float = 0.1, output_dir: str = None):
    """
    Plot pairplot on a sample of data to avoid overload.
    """
    sample_df = df[numeric_cols].sample(frac=sample_frac, random_state=42)
    sns.pairplot(sample_df)
    if output_dir:
        plt.savefig(f"{output_dir}/pairplot_sample.png")
    plt.show()

def cramers_v(x, y):
    """Calculate Cramér's V for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1))/(n - 1))
    rcorr = r - ((r - 1)**2)/(n - 1)
    kcorr = k - ((k - 1)**2)/(n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def plot_cramers_v_heatmap(df: pd.DataFrame, categorical_cols: list, output_dir: str = None):
    """
    Plot heatmap for Cramér's V correlation matrix between categorical variables.
    """
    cramers_results = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
    for col1 in categorical_cols:
        for col2 in categorical_cols:
            if col1 == col2:
                cramers_results.loc[col1, col2] = 1.0
            else:
                cramers_results.loc[col1, col2] = cramers_v(df[col1], df[col2])

    cramers_results = cramers_results.astype(float)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cramers_results, annot=True, cmap="Purples", fmt=".2f")
    plt.title("Cramér's V Correlation Between Categorical Features")
    plt.tight_layout()
    if output_dir:
        plt.savefig(f"{output_dir}/cramers_v_heatmap.png")
    plt.show()
