# train_classifier.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

# === Configuration ===
DELTA_FEATURES = [
    "delta_score_2gram", "delta_entropy_2gram", "delta_variance_2gram",
    "delta_score_3gram", "delta_entropy_3gram", "delta_variance_3gram",
    "delta_score_4gram", "delta_entropy_4gram", "delta_variance_4gram",
    "delta_score_5gram", "delta_entropy_5gram", "delta_variance_5gram",
]

TEMP_LIST = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
DATASETS = ["pubmed", "writing", "xsum"]
VARIANTS = ["gpt-4", "gemini", "turbo_3.5"]

# Input base folder where features CSVs live
INPUT_BASE = "../N-gram Scoring"
# Output base folder for evaluation results and figures
OUTPUT_BASE = "../results"

if __name__ == '__main__':
    # ensure output base exists
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    for dataset in DATASETS:
        for variant in VARIANTS:
            # create per-dataset output subfolder
            out_folder = os.path.join(OUTPUT_BASE, dataset)
            os.makedirs(out_folder, exist_ok=True)
            print(f"\n=== Evaluating: {dataset.upper()} | Variant: {variant.upper()} ===")
            results = []

            for temp in TEMP_LIST:
                # input feature CSV path
                infile = f"kenlm_filtered_features_{dataset}_{variant}_temp{temp}.csv"
                file_path = os.path.join(INPUT_BASE, infile)

                if not os.path.exists(file_path):
                    print(f"❌ Missing file: {file_path}")
                    continue

                print(f"\n→ Temp: {temp}")
                df = pd.read_csv(file_path)
                df["label_num"] = df["label"].map({"human": 0, "ai": 1})

                # features and labels
                X = df[DELTA_FEATURES]
                y = df["label_num"]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # XGBoost only
                xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
                xgb.fit(X_train, y_train)
                preds = xgb.predict(X_test)
                probs = xgb.predict_proba(X_test)[:, 1]

                auc = roc_auc_score(y_test, probs)
                acc = accuracy_score(y_test, preds)

                results.append({
                    "Temperature": temp,
                    "XGB_AUC": auc,
                    "XGB_Accuracy": acc
                })

                # plot confusion matrix
                cm = confusion_matrix(y_test, preds)
                plt.figure(figsize=(5, 4))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Human", "AI"], yticklabels=["Human", "AI"]
                )
                plt.title(f"CM: {dataset.upper()} {variant.upper()} @Temp {temp}")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                cm_file = os.path.join(out_folder, f"cm_{dataset}_{variant}_temp{temp}.png")
                plt.tight_layout()
                plt.savefig(cm_file)
                plt.close()

            # save summary CSV
            if results:
                summary_df = pd.DataFrame(results)
                summary_file = os.path.join(
                    out_folder,
                    f"temp_model_summary_{dataset}_{variant}.csv"
                )
                summary_df.to_csv(summary_file, index=False)
                print(f"🗄️ Saved summary: {summary_file}")

    print("\n✅ All XGB evaluations completed.")

