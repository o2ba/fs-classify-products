import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "data/ClassifyProducts.csv"

def load_preprocess():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id"])
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["target"])
    return df, le

def split_data(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    df, le = load_preprocess()
    X_train, X_test, y_train, y_test = split_data(df)

    X_train_bal, y_train_bal = balance_data(X_train, y_train)

    X_train_scaled, X_test_scaled = scale_data(X_train_bal, X_test)

    # Train Random Forest baseline
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train_bal)

    # Predict on test set
    y_pred = clf.predict(X_test_scaled)

    # Evaluate
    print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
