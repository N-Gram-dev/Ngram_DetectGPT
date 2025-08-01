import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# === Configuration ===
INPUT_BASE = "../N-gram Scoring"
DATASETS = ["pubmed", "writing", "xsum"]
VARIANTS = ["gpt-4", "gemini", "turbo_3.5"]
TEMP_LIST = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

# Feature subsets for ablation
FEATURE_SUBSETS = {
    "Only_L": ["delta_score_2gram","delta_score_3gram","delta_score_4gram","delta_score_5gram"],
    "Only_H": ["delta_entropy_2gram","delta_entropy_3gram","delta_entropy_4gram","delta_entropy_5gram"],
    "Only_V": ["delta_variance_2gram","delta_variance_3gram","delta_variance_4gram","delta_variance_5gram"],
    "L+H": ["delta_score_2gram","delta_score_3gram","delta_score_4gram","delta_score_5gram",
             "delta_entropy_2gram","delta_entropy_3gram","delta_entropy_4gram","delta_entropy_5gram"],
    "L+V": ["delta_score_2gram","delta_score_3gram","delta_score_4gram","delta_score_5gram",
             "delta_variance_2gram","delta_variance_3gram","delta_variance_4gram","delta_variance_5gram"],
    "H+V": ["delta_entropy_2gram","delta_entropy_3gram","delta_entropy_4gram","delta_entropy_5gram",
             "delta_variance_2gram","delta_variance_3gram","delta_variance_4gram","delta_variance_5gram"],
    "All": [
        "delta_score_2gram","delta_entropy_2gram","delta_variance_2gram",
        "delta_score_3gram","delta_entropy_3gram","delta_variance_3gram",
        "delta_score_4gram","delta_entropy_4gram","delta_variance_4gram",
        "delta_score_5gram","delta_entropy_5gram","delta_variance_5gram",
    ]
}

# Collect results
results = []
for dataset in DATASETS:
    for variant in VARIANTS:
        # load all temps into one DF
        data_frames = []
        for temp in TEMP_LIST:
            fname = f"kenlm_filtered_features_{dataset}_{variant}_temp{temp}.csv"
            path = os.path.join(INPUT_BASE, fname)
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['label_num'] = df['label'].map({'human':0,'ai':1})
                data_frames.append(df)
        if not data_frames:
            continue
        merged = pd.concat(data_frames, ignore_index=True)
        X_full = merged  # placeholder
        y = merged['label_num']
        for name, features in FEATURE_SUBSETS.items():
            X = merged[features]
            # quick train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
            model.fit(X_train, y_train)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
            results.append({
                'Dataset_Variant': f"{dataset}_{variant}",
                'Subset': name,
                'AUC': auc
            })

# Create DataFrame
df_res = pd.DataFrame(results)
# Pivot for heatmap
pivot = df_res.pivot(index='Subset', columns='Dataset_Variant', values='AUC')

# Plot heatmap
sns.set(font_scale=1.0)
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label':'ROC-AUC'})
plt.title('Ablation Study: ROC-AUC by Feature Subset and Dataset/Variant')
plt.ylabel('Feature Subset')
plt.xlabel('Dataset & Variant')
plt.tight_layout()
# Save
out_dir = os.path.join('../results','aggregated')
plt.savefig(os.path.join(out_dir,'ablation_heatmap.png'))
plt.close()
print(f"Ablation heatmap saved to {out_dir}/ablation_heatmap.png")
