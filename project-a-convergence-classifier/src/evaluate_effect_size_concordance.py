import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from align_datasets import align_three_datasets


def compute_log_fold_change(X_raw, y_binary):
    """Calculates Mean Log-Fold Change (Disease vs Control) per gene."""
    ctrl_mask = (y_binary == 0)
    dis_mask = (y_binary == 1)
    
    mean_ctrl = X_raw.loc[:, ctrl_mask].mean(axis=1)
    mean_dis = X_raw.loc[:, dis_mask].mean(axis=1)
    
    lfc = mean_dis - mean_ctrl
    return lfc


def run_concordance_analysis():
    print("\n==================================================")
    print("=== PAIRWISE LOG-FOLD-CHANGE CONCORDANCE ===")
    print("==================================================")
    
    paths = {
        "GSE18732": "project-a-convergence-classifier/data/GSE18732_series_matrix.txt",
        "GSE10946": "project-a-convergence-classifier/data/GSE10946_series_matrix.txt",
        "GSE89632": "project-a-convergence-classifier/data/GSE89632_series_matrix.txt"
    }
    biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
    gpl570 = "project-a-convergence-classifier/data/GPL570.annot"
    gpl14951 = "project-a-convergence-classifier/data/GPL14951.annot"

    # Get raw Log2 matrices
    matrices_raw, labels_raw, _ = align_three_datasets(
        paths, biomart_path, gpl570, gpl14951, standardise=False
    )
    
    lfcs = {}
    
    # Calculate Log-Fold-Change vectors across 10,369 genes
    for name, X_df in matrices_raw.items():
        if name == "GSE18732":
            mask = labels_raw[name].isin(["NGT", "DM"])
            y_bin = (labels_raw[name][mask] == "DM").astype(int)
        elif name == "GSE10946":
            mask = labels_raw[name].isin(["Control", "PCOS"])
            y_bin = (labels_raw[name][mask] == "PCOS").astype(int)
        elif name == "GSE89632":
            mask = labels_raw[name].isin(["HC", "NASH"])
            y_bin = (labels_raw[name][mask] == "NASH").astype(int)
            
        lfcs[name] = compute_log_fold_change(X_df.loc[:, mask], y_bin)

    # Correlate LFC vectors pairwise
    datasets = ["GSE18732", "GSE10946", "GSE89632"]
    pairs = [("GSE18732", "GSE10946"), ("GSE18732", "GSE89632"), ("GSE10946", "GSE89632")]
    
    results = []
    for d1, d2 in pairs:
        r_pearson, p_pearson = pearsonr(lfcs[d1], lfcs[d2])
        r_spearman, p_spearman = spearmanr(lfcs[d1], lfcs[d2])
        
        results.append({
            "Pair": f"{d1} vs {d2}",
            "Pearson r": round(r_pearson, 4),
            "Pearson p": f"{p_pearson:.4e}",
            "Spearman r": round(r_spearman, 4),
            "Spearman p": f"{p_spearman:.4e}"
        })

    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    run_concordance_analysis()