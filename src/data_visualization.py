from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    # Set a clean visual theme for plots
    sns.set_theme(style="whitegrid")

    # 1. Reliably resolve path to creditcard.csv relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "creditcard.csv"

    print(f"Loading dataset from: {data_path}...")
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        print("Please place 'creditcard.csv' inside 'data/raw/'.")
        return

    df = pd.read_csv(data_path)
    print("Dataset loaded successfully.\n")

    # Separate legitimate and fraudulent transactions for detailed comparison
    legit_df = df[df["Class"] == 0]
    fraud_df = df[df["Class"] == 1]

    # Map class labels for more readable chart descriptions
    df_plot = df.copy()
    df_plot["Transaction Type"] = df_plot["Class"].map({0: "Legitimate", 1: "Fraudulent"})

    # Create a figure with subplots for structured viewing
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # -------------------------------------------------------------------------
    # 2. Bar Chart: Transaction Class Distribution
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    sns.countplot(
        data=df_plot,
        x="Transaction Type",
        palette={"Legitimate": "#1f77b4", "Fraudulent": "#d62728"},
        hue="Transaction Type",
        legend=False,
        ax=ax1
    )
    ax1.set_title("Class Distribution (Legitimate vs Fraudulent)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Transaction Type", fontsize=11)
    ax1.set_ylabel("Number of Transactions", fontsize=11)
    ax1.set_yscale("log")  # Using log scale due to extreme class imbalance

    # Add data annotations on top of the bars
    for p in ax1.patches:
        height = int(p.get_height())
        ax1.annotate(
            f"{height:,}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 3),
            textcoords="offset points"
        )

    # -------------------------------------------------------------------------
    # 3. Histograms: Transaction Amount Distributions (Log Scale)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(
        legit_df["Amount"],
        bins=40,
        color="#1f77b4",
        label="Legitimate",
        kde=False,
        ax=ax2
    )
    sns.histplot(
        fraud_df["Amount"],
        bins=40,
        color="#d62728",
        label="Fraudulent",
        kde=False,
        ax=ax2
    )
    ax2.set_title("Transaction Amount Distribution by Class", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Transaction Amount ($)", fontsize=11)
    ax2.set_ylabel("Frequency (Log Scale)", fontsize=11)
    ax2.set_yscale("log")  # Log scale allows seeing distribution across small and large frequencies
    ax2.legend(title="Transaction Type")

    # -------------------------------------------------------------------------
    # 4. Boxplot: Transaction Amount Distributions by Class
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :])
    sns.boxplot(
        data=df_plot,
        x="Transaction Type",
        y="Amount",
        palette={"Legitimate": "#1f77b4", "Fraudulent": "#d62728"},
        hue="Transaction Type",
        legend=False,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": "7"},
        ax=ax3
    )
    ax3.set_title("Transaction Amount Boxplot by Class (Yellow dot = Mean)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Transaction Type", fontsize=11)
    ax3.set_ylabel("Transaction Amount ($)", fontsize=11)
    ax3.set_yscale("log")  # Log scale helps visualize median, IQR, and outliers clearly

    # -------------------------------------------------------------------------
    # 5. Display the Visualizations
    # -------------------------------------------------------------------------
    plt.show()


if __name__ == "__main__":
    main()
