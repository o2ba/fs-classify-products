import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # For loading saved model, optional


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def print_classification_report(y_true, y_pred, labels):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    importances = model.feature_importances_
    indices = importances.argsort()[-top_n:][::-1]  # top_n features
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    DATA_PATH = "data/ClassifyProducts.csv"

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id"])
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["target"])

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train_bal)

    y_pred = clf.predict(X_test_scaled)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print_classification_report(y_test, y_pred, le.classes_)

    plot_confusion_matrix(y_test, y_pred, le.classes_, save_path="outputs/confusion_matrix.png")

    plot_feature_importance(clf, X.columns, top_n=20, save_path="outputs/feature_importance.png")
