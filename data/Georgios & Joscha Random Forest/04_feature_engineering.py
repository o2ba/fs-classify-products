# 04_feature_engineering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "data/ClassifyProducts.csv"
OUTPUT_PATH = "outputs/"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id"])
    return df

def encode_target(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["target"])
    return df

def check_low_variance(df, threshold=0.01):
    feature_vars = df.drop(columns=["target"]).var()
    low_var_features = feature_vars[feature_vars < threshold].index.tolist()
    print(f"Low variance features (threshold={threshold}): {len(low_var_features)}")
    return low_var_features

def plot_correlation_heatmap(df):
    corr = df.drop(columns=["target"]).corr().abs()

    # Filter strong correlations
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    strong_corr_pairs = [(col, row) for col in upper.columns for row in upper.index if upper.loc[row, col] > 0.95]
    print(f"Highly correlated feature pairs (> 0.95): {len(strong_corr_pairs)}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", xticklabels=False, yticklabels=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "correlation_heatmap.png"))
    plt.close()

    return strong_corr_pairs

if __name__ == "__main__":
    df = load_data()
    df = encode_target(df)

    low_var_feats = check_low_variance(df)
    high_corr_pairs = plot_correlation_heatmap(df)
