import warnings
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from align_datasets import align_three_datasets
from clean_geo import standardise_per_gene

warnings.filterwarnings("ignore")


def parse_log_homa(file_path):
    sample_ids, homa_vals = [], []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [item.strip().strip('"') for item in line.split("\t")[1:]]
            if line.startswith("!Sample_characteristics_ch1") and '"log_homa1_ir:' in line:
                for item in line.split("\t")[1:]:
                    val_str = item.strip().strip('"').replace("log_homa1_ir:", "").strip()
                    try:
                        homa_vals.append(float(val_str))
                    except ValueError:
                        homa_vals.append(np.nan)
                break
    return pd.Series(homa_vals, index=sample_ids, name="log_homa1_ir")


def run_honest_homa_regression():
    print("\n==================================================")
    print("=== HONEST CONTINUOUS HOMA-IR REGRESSION ===")
    print("=== (FAST PARALLELIZED PERMUTATION NULL) ===")
    print("==================================================")
    
    data_path = "project-a-convergence-classifier/data/GSE18732_series_matrix.txt"
    paths = {
        "GSE18732": data_path,
        "GSE10946": "project-a-convergence-classifier/data/GSE10946_series_matrix.txt",
        "GSE89632": "project-a-convergence-classifier/data/GSE89632_series_matrix.txt"
    }
    biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
    gpl570 = "project-a-convergence-classifier/data/GPL570.annot"
    gpl14951 = "project-a-convergence-classifier/data/GPL14951.annot"

    matrices_raw, _, _ = align_three_datasets(paths, biomart_path, gpl570, gpl14951, standardise=False)
    
    X_mus_std = standardise_per_gene(matrices_raw["GSE18732"], step_label="GSE18732 Per-Gene Std").T
    s_homa = parse_log_homa(data_path)
    
    df_data = pd.concat([X_mus_std, s_homa], axis=1).dropna(subset=["log_homa1_ir"])
    X_full = df_data.drop(columns=["log_homa1_ir"])
    y = df_data["log_homa1_ir"]
    
    # Feature Filter: Keep top 2,000 variable genes to speed up computation without leakage
    top_var_genes = X_full.var(axis=0).nlargest(2000).index
    X = X_full[top_var_genes]
    
    print(f"Continuous Cohort: {X.shape[0]} samples across {X.shape[1]} top-variance genes.")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y))
    alphas = np.logspace(-1, 4, 10)
    
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        
        reg = RidgeCV(alphas=alphas, cv=3)
        reg.fit(X_train, y_train)
        oof_predictions[test_idx] = reg.predict(X_test)

    obs_r, obs_p = pearsonr(y.values, oof_predictions)
    print(f"\n---> POOLED OOF PEARSON CORRELATION: r = {obs_r:.3f} (p = {obs_p:.4e}) <---")
    
    print("\nRunning 200-Run Fast Permutation Null Test...")
    null_r_vals = []
    np.random.seed(42)
    
    for i in range(1, 201):
        y_shuffled = np.random.permutation(y.values)
        null_oof = np.zeros(len(y))
        
        for train_idx, test_idx in kf.split(X, y_shuffled):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_shuf = y_shuffled[train_idx]
            
            reg_null = RidgeCV(alphas=alphas, cv=3)
            reg_null.fit(X_train, y_train_shuf)
            null_oof[test_idx] = reg_null.predict(X_test)
            
        r_null, _ = pearsonr(y_shuffled, null_oof)
        null_r_vals.append(r_null)
        
        if i % 50 == 0:
            print(f"  [Progress] Completed {i}/200 Permutations...")
            
    p_empirical = np.mean(np.array(null_r_vals) >= obs_r)
    print("\n==================================================")
    print("=== FINAL CONTINUOUS HOMA REGRESSION SUMMARY ===")
    print("==================================================")
    print(f"Observed OOF r:      {obs_r:.3f}")
    print(f"Null Mean r:         {np.mean(null_r_vals):.3f}")
    print(f"Null 95th Pct r:     {np.percentile(null_r_vals, 95):.3f}")
    print(f"Empirical p-value:   {p_empirical:.4f}")


if __name__ == "__main__":
    run_honest_homa_regression()i