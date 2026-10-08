"""
Stream Simulator Layer
----------------------
In this project, a "stream" represents a sequential, real-time arrival of transactions.
Unlike static batch evaluation where all test data is available simultaneously,
real-world fraud detection systems ingest transactions one-by-one as they occur in time.

Since we are working with an offline benchmark dataset (creditcard.csv):
1. We preserve the strict chronological order of events using the 'Time' column (never shuffling).
2. We partition the data:
   - First 80%: Historical / training baseline data.
   - Last 20%: Streaming / test data simulating incoming future transactions.
3. We implement a generator (using Python's `yield`) that emits one transaction at a time,
   providing:
   - original dataset index
   - Time
   - feature values (V1-V28, Amount)
   - true Class label (0 = Legitimate, 1 = Fraud)

This module will serve as the input feed for downstream prediction, drift detection,
and monitoring components.
"""

from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple
import pandas as pd


def load_dataset(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the credit card transaction dataset and verifies chronological ordering.

    Parameters:
        file_path: Optional Path to creditcard.csv. If None, resolves to data/raw/creditcard.csv.

    Returns:
        pd.DataFrame: Loaded dataset sorted by 'Time' with original index preserved.
    """
    if file_path is None:
        project_root = Path(__file__).resolve().parent.parent
        file_path = project_root / "data" / "raw" / "creditcard.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}. Please place 'creditcard.csv' inside 'data/raw/'."
        )

    df = pd.read_csv(file_path)

    # Ensure chronological order based on the 'Time' column without shuffling
    df = df.sort_values(by="Time", kind="mergesort").reset_index(names="original_index")
    return df


def split_chronological(
    df: pd.DataFrame, train_ratio: float = 0.80
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset chronologically into historical and streaming portions.

    Parameters:
        df: Input DataFrame sorted by Time.
        train_ratio: Proportion of data allocated for historical training (default 0.80).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, stream_df)
    """
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].copy()
    stream_df = df.iloc[split_index:].copy()
    return train_df, stream_df


class StreamSimulator:
    """
    Simulates a real-time transaction stream by yielding transactions sequentially.
    
    Reusable across the prediction engine, drift detectors, and evaluation layers.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the stream simulator with a DataFrame.

        Parameters:
            data: DataFrame containing transactions to stream.
        """
        self.data = data
        self.feature_columns = [
            col for col in data.columns if col not in ["Class", "original_index"]
        ]

    def __len__(self) -> int:
        """Returns the total number of transactions in the stream."""
        return len(self.data)

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        """Allows direct iteration over the StreamSimulator instance."""
        return self.stream()

    def stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generator function yielding one transaction dictionary at a time.

        Yields:
            dict: {
                'index': int (original dataset row index),
                'time': float (transaction timestamp in seconds),
                'features': dict (feature column names and their respective values),
                'label': int (ground-truth class: 0 for legitimate, 1 for fraud)
            }
        """
        for _, row in self.data.iterrows():
            orig_idx = int(row["original_index"]) if "original_index" in row else int(row.name)
            time_val = float(row["Time"])
            label = int(row["Class"])
            features = {col: row[col] for col in self.feature_columns}

            yield {
                "index": orig_idx,
                "time": time_val,
                "features": features,
                "label": label,
            }


def main():
    print("=" * 60)
    print("STREAM SIMULATOR DEMONSTRATION")
    print("=" * 60)

    # 1. Load dataset
    print("Loading dataset...")
    df = load_dataset()
    print(f"Total dataset records: {len(df):,}")

    # 2. Split chronologically (80% historical, 20% stream)
    train_df, stream_df = split_chronological(df, train_ratio=0.80)
    print(f"Historical / Training set size: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Streaming / Test set size    : {len(stream_df):,} ({len(stream_df)/len(df)*100:.1f}%)\n")

    # 3. Initialize the stream simulator
    simulator = StreamSimulator(stream_df)
    total_stream_tx = len(simulator)
    print(f"Total transactions in stream: {total_stream_tx:,}\n")

    # 4. Stream and print the first 5 transactions
    print("-" * 60)
    print("First 5 Streamed Transactions:")
    print("-" * 60)

    stream_generator = simulator.stream()
    for i in range(5):
        tx = next(stream_generator)
        print(f"\n[Transaction #{i + 1}]")
        print(f"  - Original Index : {tx['index']}")
        print(f"  - Time           : {tx['time']:.1f}s")
        print(f"  - Class (Label)  : {tx['label']} ({'Fraud' if tx['label'] == 1 else 'Legitimate'})")
        # Display sample features (Time, Amount, and first 3 PCA features)
        sample_feats = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in list(tx["features"].items())[:5]
        }
        print(f"  - Sample Features: {sample_feats} ... ({len(tx['features'])} total features)")

    print("\n" + "=" * 60)
    print("Stream simulation demo completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
