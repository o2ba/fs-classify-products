# 05_data_balancing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
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

def balance_train_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

if __name__ == "__main__":
    df, le = load_and_preprocess()
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Before balancing, training set class distribution:\n{y_train.value_counts()}")

    X_train_bal, y_train_bal = balance_train_data(X_train, y_train)

    print(f"After balancing, training set class distribution:\n{y_train_bal.value_counts()}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # Optionally save the balanced training data for inspection
    train_balanced = X_train_bal.copy()
    train_balanced["target"] = y_train_bal
    train_balanced.to_csv(os.path.join(OUTPUT_PATH, "train_balanced.csv"), index=False)
