"""Visualization utilities for customer segmentation analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_demographics(df: pd.DataFrame, output_dir: str) -> None:
    """Create demographic visualizations for gender and age."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(x="Genre", data=df, palette="Set2", ax=axes[0])
    axes[0].set_title("Customer Gender Distribution")
    axes[0].set_xlabel("Gender")
    axes[0].set_ylabel("Count")

    sns.histplot(df["Age"], kde=True, color="#4c72b0", bins=12, ax=axes[1])
    axes[1].set_title("Customer Age Distribution")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Frequency")

    figure.tight_layout()
    figure.savefig(Path(output_dir) / "demographics.png", dpi=300)
    plt.close(figure)


def plot_income_spending_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Plot the income and spending score distributions."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Annual Income (k$)"], kde=True, color="#2ca02c", bins=12, ax=axes[0])
    axes[0].set_title("Annual Income Distribution (k$)")
    axes[0].set_xlabel("Annual Income (k$)")
    axes[0].set_ylabel("Frequency")

    sns.histplot(df["Spending Score (1-100)"], kde=True, color="#d62728", bins=12, ax=axes[1])
    axes[1].set_title("Spending Score Distribution")
    axes[1].set_xlabel("Spending Score")
    axes[1].set_ylabel("Frequency")

    figure.tight_layout()
    figure.savefig(Path(output_dir) / "income_spending_distribution.png", dpi=300)
    plt.close(figure)


def plot_cluster_scatter(df: pd.DataFrame, output_path: str) -> None:
    """Save an income vs spending scatter plot by cluster."""
    figure, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Customer_Segment",
        palette="tab10",
        s=80,
        alpha=0.85,
        ax=ax,
        edgecolor="w",
    )
    ax.set_title("Income vs Spending Score by Cluster")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend(title="Segment", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_cluster_distribution(df: pd.DataFrame, output_path: str) -> None:
    """Save a distribution chart for the number of customers per cluster."""
    figure, ax = plt.subplots(figsize=(8, 6))
    cluster_counts = df["Customer_Segment"].value_counts().reset_index()
    cluster_counts.columns = ["Customer_Segment", "Count"]
    sns.barplot(
        data=cluster_counts,
        x="Customer_Segment",
        y="Count",
        palette="Set3",
        ax=ax,
    )
    ax.set_title("Customer Cluster Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
