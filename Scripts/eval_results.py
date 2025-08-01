import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Configuration ===
DATASETS = ["pubmed", "writing", "xsum"]
VARIANTS = ["gpt-4", "gemini", "turbo_3.5"]
OUTPUT_BASE = "../results"
AGG_DIR = os.path.join(OUTPUT_BASE, "aggregated")

if __name__ == '__main__':
    os.makedirs(AGG_DIR, exist_ok=True)

    # Collect all summaries
    all_summaries = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            summary_file = os.path.join(
                OUTPUT_BASE,
                dataset,
                f"temp_model_summary_{dataset}_{variant}.csv"
            )
            if not os.path.exists(summary_file):
                print(f"⚠️ Missing summary: {summary_file}")
                continue
            df = pd.read_csv(summary_file)
            df["dataset"] = dataset
            df["variant"] = variant
            all_summaries.append(df)

    # Combine into one DataFrame
    if not all_summaries:
        print("No summaries found. Exiting.")
        exit(1)

    df_all = pd.concat(all_summaries, ignore_index=True)

    # Save combined CSV
    combined_csv = os.path.join(AGG_DIR, "combined_temp_summary.csv")
    df_all.to_csv(combined_csv, index=False)
    print(f"🗄️ Combined summary saved to {combined_csv}")

    # Plot AUC vs Temperature for each variant and dataset
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_all,
        x="Temperature",
        y="XGB_AUC",
        hue="variant",
        style="dataset",
        markers=True,
        dashes=False
    )
    plt.title("XGB ROC-AUC vs Temperature\nby Dataset and Variant")
    plt.tight_layout()
    plot_file = os.path.join(AGG_DIR, "auc_vs_temperature.png")
    plt.savefig(plot_file)
    print(f"📈 Plot saved to {plot_file}")
