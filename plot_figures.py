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
    # ensure aggregated dir exists
    if not os.path.exists(AGG_DIR):
        raise FileNotFoundError(f"Aggregated folder not found: {AGG_DIR}")

    # Load combined summary
    combined_csv = os.path.join(AGG_DIR, "combined_temp_summary.csv")
    df_all = pd.read_csv(combined_csv)

    sns.set(style="whitegrid")

    # Plot 1: AUC vs Temperature per dataset
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df_all,
        x="Temperature",
        y="XGB_AUC",
        hue="variant",
        style="dataset",
        markers=True,
        dashes=False
    )
    plt.title("XGB ROC-AUC vs Temperature by Dataset and Variant")
    plt.tight_layout()
    plt.savefig(os.path.join(AGG_DIR, "auc_vs_temp_combined.png"))
    plt.close()

    # Plot 2: Accuracy vs Temperature per dataset
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df_all,
        x="Temperature",
        y="XGB_Accuracy",
        hue="variant",
        style="dataset",
        markers=True,
        dashes=False
    )
    plt.title("XGB Accuracy vs Temperature by Dataset and Variant")
    plt.tight_layout()
    plt.savefig(os.path.join(AGG_DIR, "accuracy_vs_temp_combined.png"))
    plt.close()

    # Plot 3: Boxplot of AUC distribution across variants
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df_all,
        x="variant",
        y="XGB_AUC",
        palette="pastel"
    )
    plt.title("Distribution of XGB AUC across Variants")
    plt.tight_layout()
    plt.savefig(os.path.join(AGG_DIR, "auc_distribution_by_variant.png"))
    plt.close()

    print("✅ All figures generated in aggregated folder.")
