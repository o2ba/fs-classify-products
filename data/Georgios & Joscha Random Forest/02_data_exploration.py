# 02_data_exploration.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

DATA_PATH = "data/ClassifyProducts.csv"

def load_data(path):
    return pd.read_csv(path)

def check_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print("Missing values per feature:")
        print(missing.sort_values(ascending=False))

def describe_data(df):
    print("\nGeneral Info:")
    print(df.info())

    print("\nStatistical Summary:")
    print(df.describe())

def check_constant_features(df):
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"\nConstant features (only one unique value): {len(constant_cols)}")
        print(constant_cols)
    else:
        print("\nNo constant features found.")

def target_distribution(df):
    print("\nTarget distribution:")
    print(df['target'].value_counts())

    # Optional: barplot
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='target', order=df['target'].value_counts().index)
    plt.title("Target Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/target_distribution.png")
    plt.close()

if __name__ == "__main__":
    df = load_data(DATA_PATH)

    describe_data(df)
    check_missing_values(df)
    check_constant_features(df)
    target_distribution(df)
