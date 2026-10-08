"""
Rolling Stream Monitor Layer
----------------------------
This module monitors the performance of the static baseline model over consecutive
chronological windows of streaming transactions.

Key Concepts:
1. What is a Monitoring Window?
   A monitoring window is a fixed batch of consecutive transactions (e.g., 1,000 transactions)
   observed as they arrive in real-time. Instead of computing one single metric over the entire
   lifetime of the stream (which can hide localized performance drops), we evaluate performance
   window by window.

2. Why Calculate Metrics Separately for Each Window?
   In a production environment, fraud patterns evolve and data distributions shift over time
   (concept drift). Cumulative metrics average out performance over all time, masking recent
   failures. Rolling window evaluation allows us to see how model precision, recall, and false
   positives fluctuate across different time periods.

3. How Confusion Matrix Values are Updated:
   For every transaction within the current window:
   - TP (True Positive) : Actual = 1 (Fraud), Predicted = 1 (Fraud)
   - TN (True Negative) : Actual = 0 (Legit), Predicted = 0 (Legit)
   - FP (False Positive): Actual = 0 (Legit), Predicted = 1 (Fraud) -> False Alarm
   - FN (False Negative): Actual = 1 (Fraud), Predicted = 0 (Legit) -> Missed Fraud
   When the window reaches capacity (or at the end of the stream), we compute and record
   window-level metrics (Precision, Recall, F1, FPR, Accuracy) and reset counters for the next window.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
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
    Trains the baseline Logistic Regression pipeline on historical data (first 80%).

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

    pipeline.fit(X_train.values, y_train.values)
    print("Baseline model training complete.\n")
    return pipeline


def compute_window_metrics(
    tp: int, tn: int, fp: int, fn: int, window_size: int
) -> Dict[str, float]:
    """
    Computes classification metrics for a single window with division-by-zero safeguards.

    Parameters:
        tp: True Positives in window
        tn: True Negatives in window
        fp: False Positives in window
        fn: False Negatives in window
        window_size: Total transactions in window

    Returns:
        Dictionary of computed metrics (accuracy, precision, recall, f1, false_positive_rate).
    """
    # Accuracy = (TP + TN) / Total
    accuracy = (tp + tn) / window_size if window_size > 0 else 0.0

    # Precision = TP / (TP + FP)
    precision_denom = tp + fp
    precision = tp / precision_denom if precision_denom > 0 else 0.0

    # Recall = TP / (TP + FN)
    recall_denom = tp + fn
    recall = tp / recall_denom if recall_denom > 0 else 0.0

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_denom = precision + recall
    f1 = (2 * precision * recall) / f1_denom if f1_denom > 0 else 0.0

    # False Positive Rate (FPR) = FP / (FP + TN)
    fpr_denom = fp + tn
    fpr = fp / fpr_denom if fpr_denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }


def run_rolling_monitor(
    pipeline: Pipeline,
    simulator: StreamSimulator,
    window_size: int = 1000
) -> List[Dict[str, Any]]:
    """
    Monitors streaming transactions across consecutive fixed-size windows.

    Parameters:
        pipeline: Trained baseline scikit-learn model pipeline.
        simulator: StreamSimulator providing sequential transactions.
        window_size: Number of transactions per evaluation window (default 1000).

    Returns:
        List[Dict[str, Any]]: List of window-level result dictionaries.
    """
    window_results: List[Dict[str, Any]] = []

    # Window tracking state
    current_window_num = 1
    window_start_idx = None
    window_end_idx = None
    window_tx_count = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    feature_cols = simulator.feature_columns

    print("=" * 90)
    print("ROLLING STREAM MONITOR")
    print("=" * 90)
    print(
        f"{'Window':<7} | {'Tx Count':<8} | {'Actual Fraud':<12} | {'Pred Fraud':<10} | "
        f"{'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'FPR':<6}"
    )
    print("-" * 90)

    stream_generator = simulator.stream()

    for tx in stream_generator:
        # Mark window start index
        if window_start_idx is None:
            window_start_idx = tx["index"]
        window_end_idx = tx["index"]
        window_tx_count += 1

        # Extract features and predict using 2D array for optimal per-transaction speed
        feature_vector = [tx["features"][col] for col in feature_cols]
        proba = float(pipeline.predict_proba([feature_vector])[0, 1])
        predicted_label = 1 if proba >= 0.5 else 0
        actual_label = tx["label"]

        # Update window confusion matrix
        if actual_label == 1 and predicted_label == 1:
            tp += 1
        elif actual_label == 0 and predicted_label == 0:
            tn += 1
        elif actual_label == 0 and predicted_label == 1:
            fp += 1
        elif actual_label == 1 and predicted_label == 0:
            fn += 1

        # Check if window is full
        if window_tx_count == window_size:
            metrics = compute_window_metrics(tp, tn, fp, fn, window_tx_count)
            actual_fraud = tp + fn
            pred_fraud = tp + fp

            record = {
                "window_number": current_window_num,
                "start_index": window_start_idx,
                "end_index": window_end_idx,
                "transactions": window_tx_count,
                "actual_fraud": actual_fraud,
                "predicted_fraud": pred_fraud,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
            }
            window_results.append(record)

            print(
                f"{current_window_num:<7} | {window_tx_count:<8} | {actual_fraud:<12} | {pred_fraud:<10} | "
                f"{metrics['precision']:<9.4f} | {metrics['recall']:<6.4f} | {metrics['f1']:<6.4f} | {metrics['false_positive_rate']:<6.4f}"
            )

            # Reset state for next window
            current_window_num += 1
            window_start_idx = None
            window_end_idx = None
            window_tx_count = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0

    # Process final incomplete window (if any remaining transactions)
    if window_tx_count > 0:
        metrics = compute_window_metrics(tp, tn, fp, fn, window_tx_count)
        actual_fraud = tp + fn
        pred_fraud = tp + fp

        record = {
            "window_number": current_window_num,
            "start_index": window_start_idx,
            "end_index": window_end_idx,
            "transactions": window_tx_count,
            "actual_fraud": actual_fraud,
            "predicted_fraud": pred_fraud,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_positive_rate": metrics["false_positive_rate"],
        }
        window_results.append(record)

        print(
            f"{current_window_num:<7} | {window_tx_count:<8} | {actual_fraud:<12} | {pred_fraud:<10} | "
            f"{metrics['precision']:<9.4f} | {metrics['recall']:<6.4f} | {metrics['f1']:<6.4f} | {metrics['false_positive_rate']:<6.4f}"
        )

    print("=" * 90)
    return window_results


def main():
    # 1. Load the dataset chronologically
    print("1. Loading dataset from data/raw/creditcard.csv...")
    df = load_dataset()
    print(f"   Total dataset transactions: {len(df):,}")

    # 2. Chronological split (80% historical, 20% streaming)
    train_df, stream_df = split_chronological(df, train_ratio=0.80)
    print(f"   Historical training portion: {len(train_df):,} (80%)")
    print(f"   Streaming evaluation portion: {len(stream_df):,} (20%)\n")

    # 3. Train the baseline model on historical data only
    print("2. Training Baseline Model...")
    pipeline = train_baseline_model(train_df)

    # 4. Initialize Stream Simulator on the 20% stream portion
    print("3. Initializing Stream Simulator...")
    simulator = StreamSimulator(stream_df)
    total_stream_tx = len(simulator)
    print(f"   Available transactions in stream: {total_stream_tx:,}\n")

    # 5. Run rolling window monitoring over the entire stream
    window_size = 1000
    results = run_rolling_monitor(pipeline, simulator, window_size=window_size)

    # 6. Overall stream summary statistics
    total_processed = sum(r["transactions"] for r in results)
    total_actual_fraud = sum(r["actual_fraud"] for r in results)
    total_predicted_fraud = sum(r["predicted_fraud"] for r in results)
    total_windows = len(results)

    print("\n" + "=" * 60)
    print("OVERALL STREAM MONITORING SUMMARY")
    print("=" * 60)
    print(f"Total Transactions Processed : {total_processed:,}")
    print(f"Total Actual Fraud Cases     : {total_actual_fraud:,}")
    print(f"Total Predicted Fraud Cases  : {total_predicted_fraud:,}")
    print(f"Total Monitoring Windows     : {total_windows}")
    print("=" * 60)

    # 7. Print first and last stored window dictionaries
    if results:
        print("\n" + "=" * 60)
        print("SAMPLE STORED WINDOW RESULTS")
        print("=" * 60)
        print("\nFirst Window:")
        print(results[0])
        print("\nLast Window:")
        print(results[-1])
        print("=" * 60)


if __name__ == "__main__":
    main()

