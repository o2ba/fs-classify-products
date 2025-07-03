# 03_data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

DATA_PATH = "data/ClassifyProducts.csv"
OUTPUT_PATH = "outputs/"

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    # Drop 'id' column (IST NICHT HILFREIVHV!!!!!!!!!!!!)
    df = df.drop(columns=["id"])

    label_encoder = LabelEncoder()
    df["target"] = label_encoder.fit_transform(df["target"])

    return df, label_encoder

def split_features_labels(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def split_train_test(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def save_label_mapping(encoder):
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    pd.Series(mapping).to_csv(os.path.join(OUTPUT_PATH, "label_mapping.csv"))
    print("Saved label mapping to outputs/label_mapping.csv")

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df, encoder = preprocess(df)
    save_label_mapping(encoder)

    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print(f"Prepared dataset:")
    print(f"  Features shape: {X.shape}")
    print(f"  Train set: {X_train.shape}, Test set: {X_test.shape}")
