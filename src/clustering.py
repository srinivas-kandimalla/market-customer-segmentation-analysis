"""Customer segmentation using K-Means clustering."""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def compute_elbow(
    feature_df: pd.DataFrame,
    max_k: int,
    output_path: str,
) -> List[float]:
    """Compute K-Means inertia values for a range of cluster sizes."""
    inertias: List[float] = []
    k_values: List[int] = list(range(1, max_k + 1))

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(feature_df)
        inertias.append(model.inertia_)

    figure, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertias, marker="o", linestyle="-", color="#1f77b4")
    ax.set_title("Elbow Method for Optimal K")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    ax.grid(True, linestyle="--", alpha=0.5)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return inertias


def run_kmeans(feature_df: pd.DataFrame, n_clusters: int) -> Tuple[KMeans, np.ndarray]:
    """Fit K-Means and return the trained model and cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(feature_df)
    return model, labels


def profile_clusters(df: pd.DataFrame, cluster_column: str = "Cluster") -> pd.DataFrame:
    """Create a cluster profile summary with average metrics."""
    if cluster_column not in df.columns:
        raise KeyError(f"Cluster column '{cluster_column}' not found in dataframe.")

    summary = (
        df.groupby(cluster_column)
        .agg(
            Customer_Count=("CustomerID", "count"),
            Avg_Age=("Age", "mean"),
            Avg_Annual_Income_k=("Annual Income (k$)", "mean"),
            Avg_Annual_Income_INR=("Annual_Income_INR", "mean"),
            Avg_Spending_Score=("Spending Score (1-100)", "mean"),
        )
        .round(1)
        .reset_index()
    )
    return summary


def assign_business_labels(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Assign business-friendly cluster labels using income and spending summary."""
    summary_df = summary_df.copy()
    income_mean = summary_df["Avg_Annual_Income_k"].mean()
    spend_mean = summary_df["Avg_Spending_Score"].mean()

    def _label(row: pd.Series) -> str:
        if row["Avg_Annual_Income_k"] >= income_mean and row["Avg_Spending_Score"] >= spend_mean:
            return "Premium Customers"
        if row["Avg_Annual_Income_k"] < income_mean and row["Avg_Spending_Score"] >= spend_mean:
            return "High Spenders"
        if row["Avg_Annual_Income_k"] < income_mean and row["Avg_Spending_Score"] < spend_mean:
            return "Budget Customers"
        if row["Avg_Annual_Income_k"] >= income_mean and row["Avg_Spending_Score"] < spend_mean:
            return "Potential Customers"
        return "Regular Customers"

    summary_df["Business_Label"] = summary_df.apply(_label, axis=1)
    return summary_df


def add_cluster_labels(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    """Map cluster IDs to business-friendly labels."""
    df = df.copy()
    df["Customer_Segment"] = df["Cluster"].map(label_map)
    return df
