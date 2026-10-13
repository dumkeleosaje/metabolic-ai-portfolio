import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return mask, y_binary


def run_cross_dataset_transfer_experiment():
    print("\n==================================================")
    print("=== BUG-FIXED CROSS-DATASET TRANSFER TEST ===")
    print("==================================================")
    
    paths = {
        "GSE18732": "project-a-convergence-classifier/data/GSE18732_series_matrix.txt",
        "GSE10946": "project-a-convergence-classifier/data/GSE10946_series_matrix.txt",
        "GSE89632": "project-a-convergence-classifier/data/GSE89632_series_matrix.txt"
    }
    biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
    gpl570 = "project-a-convergence-classifier/data/GPL570.annot"
    gpl14951 = "project-a-convergence-classifier/data/GPL14951.annot"

    # FIX 1: Fetch raw Log2 matrices (standardise=False) to prevent double-scaling
    matrices_raw, labels_raw, _ = align_three_datasets(
        paths, biomart_path, gpl570, gpl14951, standardise=False
    )
    
    # Apply ONLY Per-Gene Standardisation to raw matrices
    std_matrices = {}
    binary_data = {}
    
    for name, X_df in matrices_raw.items():
        X_std = standardise_per_gene(X_df, step_label=f"{name} Raw->Per-Gene")
        mask, y_bin = get_binary_target(name, labels_raw[name])
        
        X_bin = X_std.loc[:, mask].T  # Samples x Genes
        std_matrices[name] = X_bin
        binary_data[name] = y_bin

    datasets = ["GSE18732", "GSE10946", "GSE89632"]
    transfer_pairs = [(src, tgt) for src in datasets for tgt in datasets if src != tgt]
    transfer_results = []

    for src_name, tgt_name in transfer_pairs:
        X_src, y_src = std_matrices[src_name], binary_data[src_name]
        X_tgt, y_tgt = std_matrices[tgt_name], binary_data[tgt_name]
        
        # ANOVA Feature Selection on SOURCE ONLY
        selector = SelectKBest(score_func=f_classif, k=500)
        X_src_sub = selector.fit_transform(X_src, y_src)
        X_tgt_sub = selector.transform(X_tgt)
        
        # Train ElasticNet on Source
        model = LogisticRegression(penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5, max_iter=2000, random_state=42, class_weight="balanced")
        model.fit(X_src_sub, y_src)
        
        # Predict on Target
        probs_tgt = model.predict_proba(X_tgt_sub)[:, 1]
        obs_auc = roc_auc_score(y_tgt, probs_tgt)
        
        # 200-Run Permutation Null
        n_permutations = 200
        null_aucs = []
        np.random.seed(42)
        
        for _ in range(n_permutations):
            y_tgt_shuffled = np.random.permutation(y_tgt.values)
            null_auc = roc_auc_score(y_tgt_shuffled, probs_tgt)
            null_aucs.append(null_auc)
            
        # FIX 2: Compute Two-Sided Empirical p-value
        null_array = np.array(null_aucs)
        p_greater = np.mean(null_array >= obs_auc)
        p_less = np.mean(null_array <= obs_auc)
        two_sided_p = 2 * min(p_greater, p_less)
        
        transfer_results.append({
            "Source (Train)": src_name,
            "Target (Test)": tgt_name,
            "Observed AUC": round(obs_auc, 3),
            "Null Mean AUC": round(np.mean(null_array), 3),
            "Two-Sided p-val": round(two_sided_p, 4)
        })

    print("\n==================================================")
    print("=== SUMMARY BUG-FIXED TRANSFER TABLE ===")
    print("==================================================")
    df_summary = pd.DataFrame(transfer_results)
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    run_cross_dataset_transfer_experiment()