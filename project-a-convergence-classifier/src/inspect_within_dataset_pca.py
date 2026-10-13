import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from load_geo import load_dataset
from clean_geo import clean_expression_matrix, standardise_per_sample
from aggregate_genes import aggregate_enst_to_gene


def run_gse18732_pca():
    print("=== WITHIN-DATASET PCA: GSE18732 (Skeletal Muscle) ===")
    
    # 1. Load raw matrix & target labels
    data_path = "project-a-convergence-classifier/data/GSE18732_series_matrix.txt"
    biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
    
    X_raw, y = load_dataset(data_path, target_keyword="glycemiagroup")
    print(f"Loaded GSE18732: {X_raw.shape[1]} samples across {X_raw.shape[0]} raw probes.")
    print("Class Distribution:\n", y.value_counts())
    
    # 2. Preprocess & Aggregate to Genes
    X_log2 = clean_expression_matrix(X_raw, control_prefix="AFFX-", standardise=False)
    X_genes = aggregate_enst_to_gene(X_log2, biomart_path)
    
    # 3. Per-Sample Standardisation
    X_std = standardise_per_sample(X_genes, step_label="GSE18732 Standardisation")
    
    # 4. Fit PCA (Samples x Genes: 118 x 12535)
    X_samples = X_std.T.values
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_samples)
    var_exp = pca.explained_variance_ratio_ * 100
    
    df_pca = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Glycemia_Group": y.values
    }, index=X_std.columns)
    
    # 5. Plot Within-Dataset Embedding
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_pca, x="PC1", y="PC2", hue="Glycemia_Group",
        palette={"NGT": "#2ca02c", "IGT": "#ff7f0e", "DM": "#d62728"},
        s=80, alpha=0.85
    )
    plt.title(
        f"GSE18732 Within-Dataset PCA (Skeletal Muscle)\nPC1: {var_exp[0]:.1f}% | PC2: {var_exp[1]:.1f}%",
        fontsize=12, fontweight="bold"
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Save figure
    figures_dir = "project-a-convergence-classifier/figures"
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "gse18732_within_dataset_pca.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n[SUCCESS] Figure saved to: {out_path}")
    plt.show()

if __name__ == "__main__":
    run_gse18732_pca()