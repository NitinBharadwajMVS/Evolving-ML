from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # 1. Resolve path to creditcard.csv relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "creditcard.csv"

    print(f"Loading dataset from: {data_path}...")
    if not data_path.exists():
        print(f"Error: Dataset file not found at {data_path}")
        print("Please place 'creditcard.csv' inside 'data/raw/'.")
        return

    df = pd.read_csv(data_path)
    print("Dataset loaded successfully.\n")

    # 2. Separate features (X) and target variable (y)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 3. Chronological split without shuffling (80% train, 20% test)
    # Since streaming/time-series data arrives sequentially, we do not shuffle.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )

    # 4. Print dataset split sizes and class distributions
    print("=" * 60)
    print("1. DATASET SPLIT & CLASS DISTRIBUTION")
    print("=" * 60)
    print(f"Total Transactions: {len(df):,}")
    print(f"Training Set Size : {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  - Legitimate (0): {(y_train == 0).sum():,}")
    print(f"  - Fraudulent (1): {(y_train == 1).sum():,}")
    print(f"Testing Set Size  : {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
    print(f"  - Legitimate (0): {(y_test == 0).sum():,}")
    print(f"  - Fraudulent (1): {(y_test == 1).sum():,}\n")

    # 5. Build pipeline: StandardScaler + Logistic Regression
    # Pipeline ensures standard scaling is fitted ONLY on training data, preventing data leakage.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    # 6. Train the model on training data only
    print("Training Logistic Regression baseline model...")
    pipeline.fit(X_train, y_train)
    print("Training complete.\n")

    # 7. Generate predictions and predicted probabilities on the test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of being fraudulent (Class 1)

    # 8. Evaluate and print metrics
    print("=" * 60)
    print("2. MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  TN: {cm[0, 0]:<6} | FP: {cm[0, 1]:<6}")
    print(f"  FN: {cm[1, 0]:<6} | TP: {cm[1, 1]:<6}\n")

    # Precision, Recall, F1-score for fraud class
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"Precision (Fraud - Class 1)       : {precision:.4f}")
    print(f"Recall (Fraud - Class 1)          : {recall:.4f}")
    print(f"F1-Score (Fraud - Class 1)        : {f1:.4f}")
    print(f"Average Precision (PR-AUC)        : {pr_auc:.4f}\n")

    # Full Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Fraud (1)"], digits=4))


if __name__ == "__main__":
    main()
