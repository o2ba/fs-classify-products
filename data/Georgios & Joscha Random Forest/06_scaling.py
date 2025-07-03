# 06_scaling.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

DATA_PATH = "data/ClassifyProducts.csv"
OUTPUT_PATH = "outputs/"

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id"])
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["target"])
    return df, le

def split_data(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    df, le = load_and_preprocess()
    X_train, X_test, y_train, y_test = split_data(df)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("Features scaled using StandardScaler.")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(f"{OUTPUT_PATH}X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(f"{OUTPUT_PATH}X_test_scaled.csv", index=False)

    pd.DataFrame(y_train).to_csv(f"{OUTPUT_PATH}y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{OUTPUT_PATH}y_test.csv", index=False)
