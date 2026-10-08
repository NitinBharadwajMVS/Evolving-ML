"""
Stream Predictor Layer
----------------------
This module connects our trained static Logistic Regression baseline model to the
real-time transaction stream simulator.

In real-world fraud detection:
1. A model is initially trained on historical data (offline training).
2. As new transactions arrive sequentially in production, the model must make 
   instantaneous predictions (inference) one transaction at a time.
3. This module demonstrates single-transaction inference without batching or looking ahead,
   mirroring actual streaming deployment.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
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
    - LogisticRegression: handles extreme class imbalance with class_weight='balanced'.

    Parameters:
        train_df: DataFrame containing the first 80% historical transactions.

    Returns:
        Pipeline: Trained scikit-learn pipeline.
    """
    # Exclude metadata and target columns from feature set
    cols_to_drop = [col for col in ["Class", "original_index"] if col in train_df.columns]
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df["Class"]

    print(f"Training baseline model on {len(X_train):,} historical records...")

    # Build and fit the standard pipeline
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


def predict_stream(
    pipeline: Pipeline,
    simulator: StreamSimulator,
    max_transactions: int = 1000
) -> List[Dict[str, Any]]:
    """
    Processes streaming transactions one at a time and computes predictions.

    Parameters:
        pipeline: The pre-trained scikit-learn model pipeline.
        simulator: StreamSimulator instance emitting sequential transactions.
        max_transactions: Number of stream transactions to process.

    Returns:
        List[Dict[str, Any]]: List of recorded prediction results.
    """
    results: List[Dict[str, Any]] = []
    stream_generator = simulator.stream()

    for count, tx in enumerate(stream_generator, start=1):
        if count > max_transactions:
            break

        # 1. Extract feature values as a single-row DataFrame matching training feature columns
        feature_dict = tx["features"]
        feature_df = pd.DataFrame([feature_dict])

        # 2. Compute probability of fraud (Class = 1) using predict_proba
        # predict_proba returns [[P(Class=0), P(Class=1)]]
        probabilities = pipeline.predict_proba(feature_df)[0]
        fraud_probability = float(probabilities[1])

        # 3. Predict class label using standard 0.5 decision threshold
        predicted_label = 1 if fraud_probability >= 0.5 else 0

        # 4. Record transaction details and prediction output
        record = {
            "index": tx["index"],
            "time": tx["time"],
            "actual_label": tx["label"],
            "predicted_label": predicted_label,
            "fraud_probability": fraud_probability,
        }
        results.append(record)

    return results


def main():
    print("=" * 70)
    print("STREAM PREDICTOR DEMONSTRATION")
    print("=" * 70)

    # 1. Load the dataset chronologically
    print("1. Loading dataset from data/raw/creditcard.csv...")
    df = load_dataset()
    print(f"   Total dataset transactions: {len(df):,}")

    # 2. Split into historical (80%) and streaming (20%) portions
    train_df, stream_df = split_chronological(df, train_ratio=0.80)
    print(f"   Historical training portion: {len(train_df):,} (80%)")
    print(f"   Streaming evaluation portion: {len(stream_df):,} (20%)\n")

    # 3. Train the baseline Logistic Regression model on historical portion only
    print("2. Training Baseline Model...")
    pipeline = train_baseline_model(train_df)

    # 4. Initialize Stream Simulator on the streaming portion
    print("3. Initializing Stream Simulator...")
    simulator = StreamSimulator(stream_df)
    total_available_stream = len(simulator)
    print(f"   Available transactions in stream: {total_available_stream:,}")

    # 5. Process the first 1000 transactions one-by-one
    num_to_process = 1000
    print(f"\n4. Processing first {num_to_process} streaming transactions one-by-one...\n")
    results = predict_stream(pipeline, simulator, max_transactions=num_to_process)

    # 6. Display the first 10 prediction results
    print("=" * 70)
    print("FIRST 10 PREDICTION RESULTS")
    print("=" * 70)
    print(f"{'#':<4} | {'Index':<8} | {'Time (s)':<10} | {'Actual':<8} | {'Predicted':<10} | {'Fraud Prob':<10}")
    print("-" * 70)
    for i, res in enumerate(results[:10], start=1):
        actual_str = "Fraud (1)" if res["actual_label"] == 1 else "Legit (0)"
        pred_str = "Fraud (1)" if res["predicted_label"] == 1 else "Legit (0)"
        print(
            f"{i:<4} | {res['index']:<8} | {res['time']:<10.1f} | "
            f"{actual_str:<8} | {pred_str:<10} | {res['fraud_probability']:<10.4f}"
        )

    # 7. Print summary statistics
    processed_count = len(results)
    actual_fraud_count = sum(1 for r in results if r["actual_label"] == 1)
    predicted_fraud_count = sum(1 for r in results if r["predicted_label"] == 1)

    print("\n" + "=" * 70)
    print("STREAM PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total Streamed Transactions Processed : {processed_count:,}")
    print(f"Actual Fraud Cases in Window          : {actual_fraud_count}")
    print(f"Predicted Fraud Cases in Window       : {predicted_fraud_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
