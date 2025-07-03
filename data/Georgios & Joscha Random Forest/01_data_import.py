# 01_data_import.py

import pandas as pd

DATA_PATH = "data/ClassifyProducts.csv"

def load_data(path):
    df = pd.read_csv(path)
    print("✅ Dataset loaded successfully!")
    print(f"📦 Shape: {df.shape}")
    print("\n📄 First 5 rows:")
    print(df.head())
    return df

def describe_target(df):
    print("\n🎯 Target class distribution:")
    print(df['target'].value_counts())
    print(f"\nNumber of unique target classes: {df['target'].nunique()}")

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    describe_target(df)