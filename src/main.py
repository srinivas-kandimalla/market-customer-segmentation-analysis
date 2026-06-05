"""Main entry point for the Market & Customer Segmentation Analysis project."""

from pathlib import Path

import pandas as pd

from preprocessing import prepare_customer_data, scale_features
from clustering import (
    add_cluster_labels,
    assign_business_labels,
    compute_elbow,
    profile_clusters,
    run_kmeans,
)
from pca_analysis import apply_pca, plot_pca_clusters
from visualization import (
    plot_cluster_distribution,
    plot_cluster_scatter,
    plot_demographics,
    plot_income_spending_distribution,
)


def get_segment_mapping(summary: pd.DataFrame) -> dict:
    """Create a cluster-to-business-label mapping from summary output."""
    label_map = summary.set_index("Cluster")["Business_Label"].to_dict()
    return label_map


def main() -> None:
    """Execute the end-to-end segmentation analysis workflow."""
    base_path = Path(__file__).resolve().parent.parent
    data_path = base_path / "data" / "Mall_Customers.csv"
    output_dir = base_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing dataset...")
    df = prepare_customer_data(str(data_path))

    print("Generating exploratory visualizations...")
    plot_demographics(df, str(output_dir))
    plot_income_spending_distribution(df, str(output_dir))

    print("Preparing features for clustering...")
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    scaled_df = scale_features(df, features)

    print("Running the Elbow Method to select k...")
    elbow_path = output_dir / "elbow_method.png"
    compute_elbow(scaled_df, max_k=10, output_path=str(elbow_path))

    optimal_k = 5
    print(f"Fitting K-Means with k={optimal_k}...")
    _, labels = run_kmeans(scaled_df, n_clusters=optimal_k)
    df["Cluster"] = labels

    print("Creating cluster profiles...")
    summary_df = profile_clusters(df, cluster_column="Cluster")
    labeled_summary = assign_business_labels(summary_df)
    labeled_summary = labeled_summary.rename(columns={
        "Avg_Annual_Income_INR": "Avg_Annual_Income_INR",
    })

    mapping = get_segment_mapping(labeled_summary)
    df = add_cluster_labels(df, mapping)

    print("Exporting segmented customer results...")
    df.to_csv(output_dir / "customer_segments.csv", index=False)
    labeled_summary.to_csv(output_dir / "cluster_summary.csv", index=False)

    print("Visualizing PCA and cluster charts...")
    pca_df, _ = apply_pca(scaled_df, n_components=2)
    plot_pca_clusters(pca_df, labels, str(output_dir / "pca_clusters.png"))
    plot_cluster_scatter(df, str(output_dir / "income_vs_spending_clusters.png"))
    plot_cluster_distribution(df, str(output_dir / "cluster_distribution.png"))

    print("Analysis complete. Outputs written to the outputs directory.")


if __name__ == "__main__":
    main()
