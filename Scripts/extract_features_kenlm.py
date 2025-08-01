# === Imports ===
import os
import pandas as pd
import numpy as np
import nltk
import kenlm
from collections import Counter
from scipy.stats import entropy as calc_entropy

nltk.download('punkt')

# === KenLM Model Paths ===
MODEL_DIR = '../models'
MODEL_FULL_2GRAM = os.path.join(MODEL_DIR, '2-gram.arpa.bin')
MODEL_FULL_3GRAM = os.path.join(MODEL_DIR, '3-gram.arpa.bin')
MODEL_FULL_4GRAM = os.path.join(MODEL_DIR, '4-gram.arpa.bin')
MODEL_FULL_5GRAM = os.path.join(MODEL_DIR, '5-gram.arpa.bin')

# === Load KenLM Models ===
print('Loading KenLM models...')
lm_full_2 = kenlm.Model(MODEL_FULL_2GRAM)
lm_full_3 = kenlm.Model(MODEL_FULL_3GRAM)
lm_full_4 = kenlm.Model(MODEL_FULL_4GRAM)
lm_full_5 = kenlm.Model(MODEL_FULL_5GRAM)
print('Models loaded.')

# === Utility Functions ===
def get_ngrams(text, n):
    tokens = text.lower().split()
    return [tokens[i:i + n] for i in range(len(tokens) - n + 1)]

def get_ngram_prob(model, ngram):
    phrase = " ".join(ngram)
    log_prob = model.score(phrase, bos=False, eos=False)
    return 10 ** log_prob if log_prob > float('-inf') else 0.0

def new_logic_score(model, text, n, lam=0.4):
    ngrams = get_ngrams(text, n)
    score = 0.0
    for ng in ngrams:
        phrase = " ".join(ng)
        if model.score(phrase, bos=False, eos=False) > float('-inf'):
            token_sum = 0.0
            for j in range(n):
                context = ng[max(0, j - (n - 1)):j]
                current = ng[j]
                full_ng = context + [current]
                token_sum += get_ngram_prob(model, full_ng)
            score += token_sum
        else:
            if n > 2:
                back_ng1 = ng[:-1]
                back_ng2 = ng[1:]
                p1 = get_ngram_prob(model, back_ng1) if len(back_ng1) == n - 1 else 0
                p2 = get_ngram_prob(model, back_ng2) if len(back_ng2) == n - 1 else 0
                score += (lam / 2) * (p1 + p2)
            # no contribution for bigrams when unseen
    return score

def calculate_entropy(ngrams):
    joined = [" ".join(g) for g in ngrams]
    count = Counter(joined)
    total = sum(count.values())
    probs = [v / total for v in count.values()]
    return calc_entropy(probs)

def frequency_variance(ngrams):
    joined = [" ".join(g) for g in ngrams]
    count = Counter(joined)
    values = list(count.values())
    return np.var(values) if len(values) > 1 else 0.0

# === Core Feature Extraction ===
def extract_all_features(text, sample_id=None, temp_label=None, variation_id=None):
    ngrams_2 = get_ngrams(text, 2)
    ngrams_3 = get_ngrams(text, 3)
    ngrams_4 = get_ngrams(text, 4)
    ngrams_5 = get_ngrams(text, 5)

    return {
        "score_2gram":   new_logic_score(lm_full_2, text, 2),
        "entropy_2gram": calculate_entropy(ngrams_2),
        "variance_2gram":frequency_variance(ngrams_2),
        "score_3gram":   new_logic_score(lm_full_3, text, 3),
        "entropy_3gram": calculate_entropy(ngrams_3),
        "variance_3gram":frequency_variance(ngrams_3),
        "score_4gram":   new_logic_score(lm_full_4, text, 4),
        "entropy_4gram": calculate_entropy(ngrams_4),
        "variance_4gram":frequency_variance(ngrams_4),
        "score_5gram":   new_logic_score(lm_full_5, text, 5),
        "entropy_5gram": calculate_entropy(ngrams_5),
        "variance_5gram":frequency_variance(ngrams_5),
    }

# === Feature Extraction Pipeline ===
def process_variations(df, temp_label):
    rows = []
    for i, row in df.iterrows():
        original_text = str(row["original"])
        label = row["tag"].strip().lower()
        try:
            original_feats = extract_all_features(original_text, i, temp_label, "original")
        except:
            continue

        for j in range(1, 11):
            colname = f"variant_{j}"
            if colname not in row or pd.isna(row[colname]):
                continue
            variant = str(row[colname]).strip()
            if len(variant.split()) < 3:
                continue
            try:
                variant_feats = extract_all_features(variant, i, temp_label, j)
            except:
                continue

            delta_feats = {
                f"delta_{k}": abs(original_feats[k] - variant_feats[k])
                for k in original_feats
            }
            row_data = {
                "sample_id":     i,
                "temperature":   temp_label,
                "variation_id":  j,
                "label":         label,
                "source_text":   original_text,
                "variation_text":variant,
            }
            row_data.update(delta_feats)
            rows.append(row_data)
    return pd.DataFrame(rows)

def batch_process(model_prefix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    temps = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

    for temp in temps:
        infile = f"{model_prefix}.raw_data_variants_gpt4_temp{temp}.csv"
        outfile = os.path.join(
            output_dir,
            f"kenlm_filtered_features_{model_prefix}_temp{temp}.csv"
        )

        if not os.path.exists(infile):
            print(f"❌ Missing file: {infile}")
            continue

        df = pd.read_csv(infile)
        df_feat = process_variations(df, temp)
        df_feat.to_csv(outfile, index=False)
        print(f"✅ Saved: {outfile}")

# === Run All ===
if __name__ == "__main__":
    print("\n=== Running all datasets with new scoring logic ===")

    # Single folder for all outputs:
    OUTPUT_BASE_DIR = "../N-gram Scoring"
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # List of all model prefixes:
    model_prefixes = [
        "pubmed_gpt-4",
        "writing_gpt-4",
        "xsum_gpt-4",
        "pubmed_gemini",
        "writing_gemini",
        "xsum_gemini",
        "pubmed_turbo_3.5",
        "writing_turbo_3.5",
        "xsum_turbo_3.5",
    ]

    for prefix in model_prefixes:
        batch_process(prefix, OUTPUT_BASE_DIR)

    print("\n✅ All datasets processed successfully.")
