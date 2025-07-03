import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib

DATA_PATH = "data/ClassifyProducts.csv"

def load_preprocess():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id"])
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["target"])
    return df, le

def split_balance_scale(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_bal, y_test

if __name__ == "__main__":
    df, le = load_preprocess()
    X_train, X_test, y_train, y_test = split_balance_scale(df)

    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    rand_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='f1_weighted'  # weighted F1 score because of class imbalance
    )

    rand_search.fit(X_train, y_train)

    print("Best hyperparameters:", rand_search.best_params_)

    best_rf = rand_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(best_rf, "outputs/best_rf_model.joblib")
    print("Best model saved to outputs/best_rf_model.joblib")
