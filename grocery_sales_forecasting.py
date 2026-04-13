"""
Grocery Store Sales Analysis and Forecasting
=============================================
Performs exploratory analysis on multi-store sales data and builds a
linear regression model to forecast store revenue from store attributes.
Includes a correlation heatmap and RMSE-based evaluation.

Dataset: Stores.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ── Data Loading and Exploration ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the stores dataset and print basic information."""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(df.head(10).to_string())
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def add_sales_area_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add a Sales_Area column representing sales per unit of store area."""
    df["Sales_Area"] = df["Store_Sales"] / df["Store_Area"]
    return df


# ── Filtering ─────────────────────────────────────────────────────────────────

def high_performing_stores(df: pd.DataFrame, threshold: float = 100_000) -> pd.DataFrame:
    """Return stores with sales above the specified threshold."""
    result = df[df["Store_Sales"] > threshold]
    print(f"\nStores with sales > ${threshold:,.0f}: {len(result)}")
    return result


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_sales_histogram(df: pd.DataFrame, bins: int = 20):
    """Plot the distribution of store sales."""
    plt.figure()
    plt.hist(df["Store_Sales"], bins=bins, edgecolor="black", color="steelblue")
    plt.title("Store Sales Distribution")
    plt.xlabel("Sales ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("sales_distribution.png", dpi=150)
    plt.show()
    print("Saved: sales_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Visualise the Pearson correlation matrix for all numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="jet", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Saved: correlation_heatmap.png")


# ── Modelling ─────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, label_col: str = "Store_Sales",
                     drop_cols: list = None) -> tuple:
    """Split dataframe into feature matrix and label series."""
    if drop_cols is None:
        drop_cols = ["Store ID ", "Store_Sales"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[label_col]
    return X, y


def train_linear_regression(X_train, y_train) -> LinearRegression:
    """Fit and return a LinearRegression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> float:
    """Compute RMSE on the test set and print results."""
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mean_sales = y_test.mean()
    print(f"\nTest RMSE      : ${rmse:,.2f}")
    print(f"Mean Sales     : ${mean_sales:,.2f}")
    print(f"RMSE / Mean    : {rmse / mean_sales:.2%}")
    return rmse


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "Stores.csv"

    df = load_data(DATA_PATH)
    df = add_sales_area_ratio(df)

    high_performing_stores(df)

    plot_sales_histogram(df)
    plot_correlation_heatmap(df)

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_linear_regression(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    print("\nModel Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")
