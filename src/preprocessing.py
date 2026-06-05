"""Data preprocessing utilities for customer segmentation analysis."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(data_path: str) -> pd.DataFrame:
    """Load the Mall Customers dataset from a CSV file."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(path)
    return df


def add_customer_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add a realistic Indian customer name to each row."""
    names: List[str] = [
        "Aarav Sharma", "Ananya Patel", "Rohan Mehta", "Sanya Singh",
        "Vikram Rao", "Priya Kapoor", "Arjun Gupta", "Isha Iyer",
        "Kabir Nair", "Neha Verma", "Siddharth Joshi", "Tanya Desai",
        "Dev Malhotra", "Aisha Khan", "Sameer Reddy", "Mira Nair",
        "Rahul Chatterjee", "Meera Menon", "Nikhil Bhatia", "Riya Khanna",
        "Karan Jain", "Anjali Dubey", "Tarun Shah", "Vedika Sharma",
        "Aditya Kapoor", "Pooja Sen", "Ravi Das", "Simran Kaur",
        "Amit Raj", "Nisha Yadav", "Shreya Bose", "Harsh Lamba",
        "Deepa Nair", "Rohit Dutta", "Divya Tibrewal", "Kavya Ramesh",
        "Manav Joshi", "Richa Mehta", "Pranav Singh", "Suhana Bose",
        "Yash Sharma", "Isha Sharma", "Parth Banerjee", "Nitya Agarwal",
        "Ayaan Rao", "Diya Reddy", "Aman Jain", "Pallavi Desai",
        "Kabir Khan", "Sonal Mishra", "Aryan Chopra", "Reema Sinha"
    ]

    rng = np.random.RandomState(seed=42)
    indices = rng.randint(0, len(names), size=len(df))
    df = df.copy()
    df["Customer_Name"] = [names[idx] for idx in indices]
    return df


def convert_income_to_inr(df: pd.DataFrame, exchange_rate: float = 82.5) -> pd.DataFrame:
    """Convert annual income in k$ to Indian Rupees and add a new column."""
    df = df.copy()
    if "Annual Income (k$)" not in df.columns:
        raise KeyError("Expected column 'Annual Income (k$)' not found.")

    df["Annual_Income_INR"] = (df["Annual Income (k$)"] * 1000 * exchange_rate).round(0).astype(int)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and duplicate records."""
    df = df.copy()

    if df.duplicated().any():
        df = df.drop_duplicates(ignore_index=True)

    for column in df.columns:
        if df[column].isna().any():
            if df[column].dtype == object:
                fill_value = df[column].mode().iloc[0]
            else:
                fill_value = df[column].median()
            df[column] = df[column].fillna(fill_value)

    return df


def scale_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Scale numerical features using standard scaling."""
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_columns])
    scaled_df = pd.DataFrame(scaled_values, columns=feature_columns, index=df.index)
    return scaled_df


def prepare_customer_data(data_path: str) -> pd.DataFrame:
    """Load, clean, enrich, and scale the customer dataset."""
    df = load_data(data_path)
    df = clean_data(df)
    df = add_customer_names(df)
    df = convert_income_to_inr(df)
    return df
