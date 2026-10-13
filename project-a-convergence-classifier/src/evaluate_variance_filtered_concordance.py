import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from align_datasets import align_three_datasets


def compute_lfc(X_raw, y_binary):
    ctrl = X_raw.loc[:, y_binary == 0].mean(axis=1)
    dis = X_raw.loc[:, y_binary == 1].mean(axis=1)
    return dis - ctrl


def run_filtered_concordance():
    print("\n==================================================")
    print("=== VARIANCE-FILTERED CONCORDANCE (TOP 2,000 GENES) ===")
    print("==================================================")
    
    paths = {
        "GSE18732": "project-a-convergence-classifier/data/GSE18732_series_matrix.txt",
        "GSE10946": "project-a-convergence-classifier/data/GSE10946_series_matrix.txt",
        "GSE89632": "project-a-convergence-classifier/data/GSE89632_series_matrix.txt"
    }
    biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
    gpl570 = "project-a-convergence-classifier/data/GPL570.annot"
    gpl14951 = "project-a-convergence-classifier/data/GPL14951.annot"

    matrices_raw, labels_raw, _ = align_three_datasets(paths, biomart_path, gpl570, gpl14951, standardise=False)
    
    lfcs = {}
    variances = {}
    
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
            
        X_sub = X_df.loc[:, mask]
        lfcs[name] = compute_lfc(X_sub, y_bin)
        variances[name] = X_sub.var(axis=1)

    pairs = [("GSE18732", "GSE10946"), ("GSE18732", "GSE89632"), ("GSE10946", "GSE89632")]
    results = []

    for d1, d2 in pairs:
        # Keep top 2,000 most variable genes shared across both datasets
        top_genes = (variances[d1] + variances[d2]).nlargest(2000).index
        
        lfc1 = lfcs[d1].loc[top_genes]
        lfc2 = lfcs[d2].loc[top_genes]
        
        r_obs, _ = pearsonr(lfc1, lfc2)
        rho_obs, _ = spearmanr(lfc1, lfc2)
        
        # 1,000 Gene Permutation Null
        np.random.seed(42)
        null_r = [pearsonr(lfc1, np.random.permutation(lfc2.values))[0] for _ in range(1000)]
        
        p_emp = 2 * min(np.mean(np.array(null_r) >= r_obs), np.mean(np.array(null_r) <= r_obs))
        
        results.append({
            "Pair": f"{d1} vs {d2}",
            "Top Genes": 2000,
            "Pearson r": round(r_obs, 4),
            "Spearman r": round(rho_obs, 4),
            "Empirical p-val": round(p_emp, 4)
        })

    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    run_filtered_concordance()