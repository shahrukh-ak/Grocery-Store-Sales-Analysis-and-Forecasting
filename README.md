# Grocery Store Sales Analysis and Forecasting

Analyses multi-store sales data and builds a linear regression model to forecast store revenue from store-level features. Includes distribution analysis, a correlation heatmap, and RMSE evaluation.

## Business Context

For a retail chain, understanding which store attributes drive sales enables better resource allocation and data-driven site selection for new locations. The forecasting model quantifies how store characteristics such as area and number of daily transactions relate to overall revenue.

## Dataset

`Stores.csv` contains store-level records with columns including `Store ID`, `Store_Area`, `Items_Available`, `Daily_Customer_Count`, and `Store_Sales`.

## Methodology

**Feature Engineering:** A `Sales_Area` ratio (sales per unit area) is derived as an additional descriptive metric.

**EDA:**
- Filter and report stores with sales exceeding $100,000
- Histogram of the sales distribution (20 bins)
- Pearson correlation heatmap across all numeric features

**Model:** Linear Regression trained on an 80/20 train/test split (`random_state=42`). The `Store ID` unique identifier and the target column are excluded from features.

**Evaluation:** Root Mean Squared Error (RMSE) on the test set, compared against the mean sales value to contextualise prediction error.

## Project Structure

```
04_grocery_sales_forecasting/
├── grocery_sales_forecasting.py  # Analysis and modelling script
├── requirements.txt
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Place `Stores.csv` in the same directory and run:

```bash
python grocery_sales_forecasting.py
```

Outputs: `sales_distribution.png`, `correlation_heatmap.png`, and printed RMSE metrics.
