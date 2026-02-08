from scipy.stats import pearsonr, spearmanr

def correlation_metrics(x, y):
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p
    }


def compute_dtr(df):
    df = df.copy()
    df["DTR"] = df["LST_Day"] - df["LST_Night"]
    return df


def feature_stability(series):
    mean = series.mean()
    std = series.std()
    cv = std / mean

    return {
        "mean": mean,
        "std": std,
        "cv": cv
    }
import pandas as pd

def correlation_matrix(df, method="pearson"):
    """
    Computes feature-level correlation matrix.

    method: 'pearson' or 'spearman'
    """

    numeric_df = df.select_dtypes(include=["float", "int"])

    corr = numeric_df.corr(method=method)

    return corr
