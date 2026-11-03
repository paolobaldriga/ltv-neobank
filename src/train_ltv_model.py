"""
Lifetime Value (LTV) Prediction for Digital Neobanks
====================================================

Reproducible training script.
- Scoring window philosophy: KYC + 30 days
- Target: 1-Year LTV
- Model: LightGBM with Tweedie objective

Usage:
    python src/train_ltv_model.py
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import spearmanr
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "digital_wallet_ltv_dataset.csv")
IMAGES_DIR = os.path.join(ROOT, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}\n"
            "Please download it from Kaggle and place it in the data/ folder.\n"
            "See data/README.md for instructions."
        )
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering aligned with early-lifecycle neobank LTV approach."""
    df = df.copy()

    # Engineered ratios and scores
    df["spend_to_txn_ratio"] = df["Total_Spent"] / (df["Total_Transactions"] + 1)
    df["avg_spend_per_active_day"] = df["Total_Spent"] / (df["Active_Days"] + 1)
    df["recency_score"] = 1 / (df["Last_Transaction_Days_Ago"] + 1)
    df["support_intensity"] = df["Support_Tickets_Raised"] / (df["Active_Days"] + 1)
    df["engagement_score"] = (
        df["Active_Days"] * 0.3
        + df["Loyalty_Points_Earned"] * 0.01
        + df["Referral_Count"] * 10
        + df["Customer_Satisfaction_Score"] * 5
    )
    df["high_value_txn_flag"] = (
        df["Max_Transaction_Value"] > df["Max_Transaction_Value"].quantile(0.9)
    ).astype(int)

    # Ordinal encodings
    df["income_num"] = df["Income_Level"].map({"Low": 1, "Middle": 2, "High": 3})
    df["app_usage_num"] = df["App_Usage_Frequency"].map(
        {"Monthly": 1, "Weekly": 2, "Daily": 3}
    )

    # Categorical features for LightGBM
    cat_features = [
        "Location",
        "Income_Level",
        "App_Usage_Frequency",
        "Preferred_Payment_Method",
    ]
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def compute_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    ape = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)
    mdape = np.median(ape) * 100

    n = len(y_true)
    top_k = int(0.2 * n)
    true_top = set(np.argsort(y_true)[-top_k:])
    pred_top = set(np.argsort(y_pred)[-top_k:])
    precision = len(true_top & pred_top) / top_k

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4),
        "Spearman": round(float(spearman), 4),
        "MdAPE_%": round(float(mdape), 2),
        "Precision@top20%": round(float(precision), 4),
    }


def plot_results(y_true, y_pred, importance_df, images_dir: str):
    # Predicted vs Actual
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=1)
    plt.xlabel("Actual LTV")
    plt.ylabel("Predicted LTV")
    plt.title("Predicted vs Actual LTV (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "pred_vs_actual.png"), dpi=150)
    plt.close()

    # Feature importance
    plt.figure(figsize=(9, 7))
    top = importance_df.head(15)
    sns.barplot(data=top, y="feature", x="gain", orient="h", color="steelblue")
    plt.title("Top 15 Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "feature_importance.png"), dpi=150)
    plt.close()

    # Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, color="steelblue")
    plt.title("Residual Distribution (Test Set)")
    plt.xlabel("Actual - Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "residuals.png"), dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("1-Year LTV Prediction — Digital Neobank")
    print("=" * 60)

    df = load_data(DATA_PATH)
    df = feature_engineering(df)

    cat_features = [
        "Location",
        "Income_Level",
        "App_Usage_Frequency",
        "Preferred_Payment_Method",
    ]

    feature_cols = [c for c in df.columns if c not in ["Customer_ID", "LTV"]]
    X = df[feature_cols]
    y = df["LTV"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_features
    )
    val_data = lgb.Dataset(
        X_val, label=y_val, reference=train_data, categorical_feature=cat_features
    )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    metrics_val = compute_metrics(y_val, y_pred_val)
    metrics_test = compute_metrics(y_test, y_pred_test)

    print("\n=== Validation Metrics ===")
    for k, v in metrics_val.items():
        print(f"{k:20s}: {v}")

    print("\n=== Test Metrics ===")
    for k, v in metrics_test.items():
        print(f"{k:20s}: {v}")

    importance = pd.DataFrame(
        {
            "feature": model.feature_name(),
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)

    print("\n=== Top 10 Features ===")
    print(importance.head(10).to_string(index=False))

    plot_results(y_test, y_pred_test, importance, IMAGES_DIR)

    results = {
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "top_features": importance.head(20).to_dict(orient="records"),
        "best_iteration": model.best_iteration,
        "n_samples": len(df),
    }
    results_path = os.path.join(ROOT, "model_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPlots saved to {IMAGES_DIR}/")
    print(f"Results saved to {results_path}")
    print("Done.")


if __name__ == "__main__":
    main()
