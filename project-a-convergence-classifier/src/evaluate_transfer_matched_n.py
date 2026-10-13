import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from align_datasets import align_three_datasets
from clean_geo import standardise_per_gene

warnings.filterwarnings("ignore")


def get_binary_target(dataset_name, y_series):
    if dataset_name == "GSE18732":
        mask = y_series.isin(["NGT", "DM"])
        y_binary = (y_series[mask] == "DM").astype(int)
    elif dataset_name == "GSE10946":
        mask = y_series.isin(["Control", "PCOS"])
        y_binary = (y_series[mask] == "PCOS").astype(int)
    elif dataset_name == "GSE89632":
        mask = y_series.isin(["HC", "NASH"])
        y_binary = (y_series[mask] == "NASH").astype(int)
    return mask, y_binary


def run_matched_n_and_corrected_transfer():
    print("\n==================================================")
    print("=== 1,000 PERMUTATIONS + BH MULTIPLE TESTING ===")
    print("==================================================")
    
    paths = {
        "GSE18732": "project-a-convergence-classifier/data/GSE18732_series_matrix.txt",
        "GSE10946": "project-a-convergence-classifier/data/GSE10946_series_matrix.txt",
        "GSE89632": "project-a-convergence-classifier/data/GSE89632_series_matrix.txt"
    }
    biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
    gpl570 = "project-a-convergence-classifier/data/GPL570.annot"
    gpl14951 = "project-a-convergence-classifier/data/GPL14951.annot"

    matrices_raw, labels_raw, _ = align_three_datasets(
        paths, biomart_path, gpl570, gpl14951, standardise=False
    )
    
    std_matrices, binary_data = {}, {}
    for name, X_df in matrices_raw.items():
        X_std = standardise_per_gene(X_df, step_label=f"{name} Raw->Per-Gene")
        mask, y_bin = get_binary_target(name, labels_raw[name])
        std_matrices[name] = X_std.loc[:, mask].T
        binary_data[name] = y_bin

    datasets = ["GSE18732", "GSE10946", "GSE89632"]
    transfer_pairs = [(src, tgt) for src in datasets for tgt in datasets if src != tgt]
    raw_p_vals = []
    transfer_results = []

    print("\n--- Running 6 Transfer Pairs (1,000 Permutations) ---")
    for src_name, tgt_name in transfer_pairs:
        X_src, y_src = std_matrices[src_name], binary_data[src_name]
        X_tgt, y_tgt = std_matrices[tgt_name], binary_data[tgt_name]
        
        selector = SelectKBest(score_func=f_classif, k=500)
        X_src_sub = selector.fit_transform(X_src, y_src)
        X_tgt_sub = selector.transform(X_tgt)
        
        model = LogisticRegression(penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5, max_iter=2000, random_state=42, class_weight="balanced")
        model.fit(X_src_sub, y_src)
        
        probs_tgt = model.predict_proba(X_tgt_sub)[:, 1]
        obs_auc = roc_auc_score(y_tgt, probs_tgt)
        
        # 1,000 Permutations
        n_permutations = 1000
        np.random.seed(42)
        null_aucs = [
            roc_auc_score(np.random.permutation(y_tgt.values), probs_tgt)
            for _ in range(n_permutations)
        ]
        
        null_array = np.array(null_aucs)
        two_sided_p = 2 * min(np.mean(null_array >= obs_auc), np.mean(null_array <= obs_auc))
        raw_p_vals.append(two_sided_p)
        
        transfer_results.append({
            "Source": src_name,
            "Target": tgt_name,
            "Target N": len(y_tgt),
            "Observed AUC": round(obs_auc, 3),
            "Raw p-val": round(two_sided_p, 4)
        })

    # Apply Benjamini-Hochberg Correction
    _, bh_p_vals, _, _ = multipletests(raw_p_vals, method="fdr_bh")
    for i in range(len(transfer_results)):
        transfer_results[i]["BH Adj p-val"] = round(bh_p_vals[i], 4)

    df_transfer = pd.DataFrame(transfer_results)
    print("\n==================================================")
    print("=== MULTIPLE-TESTING CORRECTED TRANSFER TABLE ===")
    print("==================================================")
    print(df_transfer.to_string(index=False))

    # --- MATCHED-N SUBSAMPLING CONTROL (Muscle Target Subsampled to N=23) ---
    print("\n==================================================")
    print("=== MATCHED-N SUBSAMPLING CONTROL (N=23) ===")
    print("==================================================")
    
    X_pcos, y_pcos = std_matrices["GSE10946"], binary_data["GSE10946"]
    X_mus, y_mus = std_matrices["GSE18732"], binary_data["GSE18732"]
    
    subsample_aucs = []
    subsample_p_vals = []
    
    np.random.seed(42)
    for _ in range(100):
        # Subsample Muscle down to N=23 with stratified sampling
        sub_idx0 = np.random.choice(np.where(y_mus == 0)[0], size=13, replace=False)
        sub_idx1 = np.random.choice(np.where(y_mus == 1)[0], size=10, replace=False)
        sub_idx = np.concatenate([sub_idx0, sub_idx1])
        
        X_mus_sub = X_mus.iloc[sub_idx]
        y_mus_sub = y_mus.iloc[sub_idx]
        
        # Train on PCOS (N=23), Test on Subsampled Muscle (N=23)
        selector = SelectKBest(score_func=f_classif, k=500)
        X_pcos_sub = selector.fit_transform(X_pcos, y_pcos)
        X_mus_sub_feat = selector.transform(X_mus_sub)
        
        model = LogisticRegression(penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5, max_iter=2000, random_state=42, class_weight="balanced")
        model.fit(X_pcos_sub, y_pcos)
        probs = model.predict_proba(X_mus_sub_feat)[:, 1]
        
        auc = roc_auc_score(y_mus_sub, probs)
        
        # Quick 200-run null for this subsample
        nulls = [roc_auc_score(np.random.permutation(y_mus_sub.values), probs) for _ in range(200)]
        p_val = 2 * min(np.mean(np.array(nulls) >= auc), np.mean(np.array(nulls) <= auc))
        
        subsample_aucs.append(auc)
        subsample_p_vals.append(p_val)

    print(f"PCOS -> Muscle Subsampled (N=23, 100 Iterations):")
    print(f"  Mean AUC:            {np.mean(subsample_aucs):.3f} +/- {np.std(subsample_aucs):.3f}")
    print(f"  Mean p-value:        {np.mean(subsample_p_vals):.4f}")
    print(f"  Fraction p < 0.05:   {np.mean(np.array(subsample_p_vals) < 0.05)*100:.1f}%")


if __name__ == "__main__":
    run_matched_n_and_corrected_transfer()