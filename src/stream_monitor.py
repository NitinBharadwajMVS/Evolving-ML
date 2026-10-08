"""
Stream Monitor Layer
--------------------
This module monitors the performance of the static baseline model in real-time
as transactions arrive from the stream simulator.

How Stream Monitoring Works:
- As each transaction arrives, we compare the model's prediction against the actual ground-truth label.
- We maintain a running count of the 4 confusion matrix outcomes:
    1. True Positive (TP) : Actual = Fraud (1), Predicted = Fraud (1)   -> Correctly detected fraud
    2. True Negative (TN) : Actual = Legit (0), Predicted = Legit (0)   -> Correctly allowed legit transaction
    3. False Positive (FP): Actual = Legit (0), Predicted = Fraud (1)   -> False alarm (Legitimate blocked)
    4. False Negative (FN): Actual = Fraud (1), Predicted = Legit (0)   -> Missed fraud (Fraud passed through)
- From these running counts, we compute cumulative performance metrics:
    Accuracy, Precision, Recall, F1 Score, and False Positive Rate (FPR), handling division-by-zero safely.
"""

import sys
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure the 'src' directory is in sys.path for direct script execution
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from stream_simulator import StreamSimulator, load_dataset, split_chronological


def train_baseline_model(train_df: pd.DataFrame) -> Pipeline:
    """
    Trains the baseline Logistic Regression pipeline on historical data.

    Matches the exact configuration from baseline_model.py:
    - StandardScaler: scales features based strictly on training distribution.
    - LogisticRegression: handles class imbalance with class_weight='balanced'.

    Parameters:
        train_df: DataFrame containing the first 80% historical transactions.

    Returns:
        Pipeline: Trained scikit-learn pipeline.
    """
    cols_to_drop = [col for col in ["Class", "original_index"] if col in train_df.columns]
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df["Class"]

    print(f"Training baseline model on {len(X_train):,} historical records...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    print("Baseline model training complete.\n")
    return pipeline


class StreamMonitor:
    """
    Tracks and computes online classification metrics for streaming transactions.
    """

    def __init__(self):
        """Initializes confusion matrix counters to zero."""
        self.tp = 0  # True Positives
        self.tn = 0  # True Negatives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.total_processed = 0

    def update(self, actual: int, predicted: int) -> None:
        """
        Updates the running confusion matrix counters for a single transaction.

        Parameters:
            actual: Ground-truth label (0 = Legitimate, 1 = Fraud)
            predicted: Model's predicted label (0 = Legitimate, 1 = Fraud)
        """
        self.total_processed += 1

        # Check all 4 confusion matrix combinations:
        if actual == 1 and predicted == 1:
            # TP: Actual Fraud correctly predicted as Fraud
            self.tp += 1
        elif actual == 0 and predicted == 0:
            # TN: Actual Legitimate correctly predicted as Legitimate
            self.tn += 1
        elif actual == 0 and predicted == 1:
            # FP: Actual Legitimate incorrectly flagged as Fraud (False Alarm)
            self.fp += 1
        elif actual == 1 and predicted == 0:
            # FN: Actual Fraud missed and predicted as Legitimate (Missed Detection)
            self.fn += 1

    def compute_metrics(self) -> Dict[str, float]:
        """
        Calculates classification metrics with safe division-by-zero handling.

        Returns:
            Dictionary containing Accuracy, Precision, Recall, F1 Score, and False Positive Rate.
        """
        total = self.total_processed

        # Accuracy = (TP + TN) / Total
        accuracy = (self.tp + self.tn) / total if total > 0 else 0.0

        # Precision = TP / (TP + FP) -> Proportion of predicted frauds that were actually fraud
        precision_denom = self.tp + self.fp
        precision = self.tp / precision_denom if precision_denom > 0 else 0.0

        # Recall = TP / (TP + FN) -> Proportion of actual frauds that were detected
        recall_denom = self.tp + self.fn
        recall = self.tp / recall_denom if recall_denom > 0 else 0.0

        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1_denom = precision + recall
        f1 = (2 * precision * recall) / f1_denom if f1_denom > 0 else 0.0

        # False Positive Rate (FPR) = FP / (FP + TN) -> Proportion of legitimate cases wrongly flagged
        fpr_denom = self.fp + self.tn
        fpr = self.fp / fpr_denom if fpr_denom > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": fpr,
        }


def main():
    print("=" * 62)
    print("STREAM MONITOR DEMONSTRATION")
    print("=" * 62)

    # 1. Load dataset chronologically
    print("1. Loading dataset from data/raw/creditcard.csv...")
    df = load_dataset()
    print(f"   Total records: {len(df):,}")

    # 2. Split into historical (80%) and streaming (20%) portions
    train_df, stream_df = split_chronological(df, train_ratio=0.80)
    print(f"   Historical training portion: {len(train_df):,} (80%)")
    print(f"   Streaming evaluation portion: {len(stream_df):,} (20%)\n")

    # 3. Train the baseline model on historical data
    print("2. Training Baseline Model...")
    pipeline = train_baseline_model(train_df)

    # 4. Initialize Stream Simulator and Stream Monitor
    print("3. Initializing Stream Simulator and Monitor...")
    simulator = StreamSimulator(stream_df)
    monitor = StreamMonitor()

    # 5. Process the first 1000 transactions one-by-one
    num_to_process = 1000
    print(f"\n4. Monitoring stream for the first {num_to_process} transactions...\n")

    stream_generator = simulator.stream()
    for count, tx in enumerate(stream_generator, start=1):
        if count > num_to_process:
            break

        # Extract features for single transaction
        feature_df = pd.DataFrame([tx["features"]])

        # Predict probability of fraud
        fraud_probability = float(pipeline.predict_proba(feature_df)[0, 1])

        # Predict class using standard 0.5 threshold
        predicted_label = 1 if fraud_probability >= 0.5 else 0
        actual_label = tx["label"]

        # Update monitoring statistics
        monitor.update(actual=actual_label, predicted=predicted_label)

    # 6. Compute final metrics
    metrics = monitor.compute_metrics()
    actual_frauds = monitor.tp + monitor.fn
    predicted_frauds = monitor.tp + monitor.fp

    # 7. Print final monitoring summary report
    print("=" * 62)
    print("STREAM MONITORING SUMMARY")
    print("=" * 62)
    print(f"Transactions Processed : {monitor.total_processed:,}")
    print(f"Actual Fraud Cases     : {actual_frauds:,}")
    print(f"Predicted Fraud Cases  : {predicted_frauds:,}")
    print("-" * 62)
    print(f"True Positives         : {monitor.tp:,}")
    print(f"True Negatives         : {monitor.tn:,}")
    print(f"False Positives        : {monitor.fp:,}")
    print(f"False Negatives        : {monitor.fn:,}")
    print("-" * 62)
    print(f"Accuracy               : {metrics['accuracy']:.4f}")
    print(f"Precision              : {metrics['precision']:.4f}")
    print(f"Recall                 : {metrics['recall']:.4f}")
    print(f"F1 Score               : {metrics['f1_score']:.4f}")
    print(f"False Positive Rate    : {metrics['false_positive_rate']:.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
