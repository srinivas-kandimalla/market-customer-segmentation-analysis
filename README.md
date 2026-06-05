Market & Customer Segmentation — Recruitment Brief

One-line summary
-----------------
Built an end-to-end customer segmentation pipeline that processes the Mall Customers dataset, performs K-Means segmentation, and produces business-ready profiles and visualizations.

Role & Skills Demonstrated
--------------------------
- Role: Data Scientist / Python Developer
- Techniques: EDA, K-Means clustering, PCA, feature scaling, cluster profiling
- Tools & libs: Python, pandas, numpy, scikit-learn, matplotlib, seaborn

What I delivered
----------------
- Cleaned and enriched dataset (`Customer_Name`, `Annual_Income_INR`).
- Reproducible K-Means pipeline with Elbow method guidance.
- PCA and cluster visualizations for stakeholder presentations.
- Exported artifacts: `customer_segments.csv`, `cluster_summary.csv`, and PNG charts.

Key results (files)
-------------------
- `outputs/customer_segments.csv` — enriched customer table with cluster labels
- `outputs/cluster_summary.csv` — per-cluster averages (age, income, spending)
- Visuals in `outputs/`: `elbow_method.png`, `pca_clusters.png`, `income_vs_spending_clusters.png`, `cluster_distribution.png`

How to run (fast)
------------------
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

Notes for reviewers
-------------------
- Reproducible: random seeds set for clustering.
- Extensible: swap `data/Mall_Customers.csv` and adapt `src/preprocessing.py` for new schemas.
- Next steps: add automated model selection, per-segment campaign recommendations, or a Jupyter notebook narrative.

Contact
-------
For questions or to request additions (notebook, alternate algorithms), open an issue or contact the project owner.
---

## At a Glance

- Purpose: Identify customer segments for targeted marketing and product strategy.
- Input: [data/Mall_Customers.csv](data/Mall_Customers.csv)
- Outputs: Visuals and CSVs in `outputs/`

## Why Use This

Segmenting customers helps you tailor messaging, optimize ad spend, and improve customer lifetime value. This repository provides a reproducible pipeline from raw CSV to business-ready customer segments.

## Highlights

- Cleaned dataset enriched with `Customer_Name` and `Annual_Income_INR`.
- EDA charts for demographics, income, and spending behavior.
- K-Means clustering guided by the Elbow Method.
- PCA visualization highlighting cluster separability.
- Human-readable cluster labels and per-cluster summary.

---

## Project Layout

```
Market_Customer_Segmentation_Analysis/
├── data/                      # source CSV(s)
│   └── Mall_Customers.csv
├── outputs/                   # generated charts + CSVs (created at runtime)
├── src/                       # modular Python code
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── pca_analysis.py
│   ├── visualization.py
│   └── main.py                # runner
├── requirements.txt
├── README.md
└── .gitignore
```

## What You’ll Find in `outputs/`

- `elbow_method.png` — Inertia curve for selecting K.
- `pca_clusters.png` — 2D PCA rendering of clusters.
- `income_vs_spending_clusters.png` — Income vs Spending scatter with segments.
- `cluster_distribution.png` — Bar chart depicting segment size.
- `customer_segments.csv` — Enriched customer records with `Cluster` and `Customer_Segment`.
- `cluster_summary.csv` — Summary metrics per segment.

---

## Quick Start (copy & paste)

1. Create & activate a Python 3.10+ virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell
# OR
.\venv\Scripts\activate      # cmd.exe
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the analysis:

```powershell
python src/main.py
```

4. Inspect results in the `outputs/` directory.

---

## Recommended Workflow for Analysts

1. Inspect `outputs/elbow_method.png` to validate cluster choice.
2. Check `outputs/pca_clusters.png` for separability.
3. Use `outputs/customer_segments.csv` to craft targeted campaigns.

---

## Tips & Notes

- The pipeline uses a seeded random state for reproducibility.
- Small Seaborn warnings may appear during plotting but do not affect outputs.
- To adapt to another dataset, update `src/preprocessing.py` column mappings.

---

If you’d like a themed Jupyter Notebook, custom labeling strategy, or additional clustering algorithms (DBSCAN, Gaussian Mixture), tell me which and I’ll add it.

— The Data Craftsman
