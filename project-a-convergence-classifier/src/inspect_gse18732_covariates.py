import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from load_geo import load_dataset, parse_metadata
from clean_geo import clean_expression_matrix, standardise_per_sample
from aggregate_genes import aggregate_enst_to_gene


def investigate_pc1_tail_and_outliers():
    print("=== INVESTIGATING GSE18732 PCA STRUCTURE & COVARIATES ===")
    
    data_path = "project-a-convergence-classifier/data/GSE18732_series_matrix.txt"
    biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
    
    # 1. Load expression data and target labels
    X_raw, y = load_dataset(data_path, target_keyword="glycemiagroup")
    
    # 2. Extract potential driving covariates from metadata
    print("\n--- Extracting Clinical Covariates ---")
    covariates = {}
    
    # Check key metadata lines for sex, age, bmi, metformin
    for kw in ["sex", "gender", "age", "bmi", "metformin"]:
        try:
            cov_map = parse_metadata(data_path, target_keyword=kw)
            covariates[kw] = cov_map
            print(f"  [FOUND] Covariate map for '{kw}' extracted successfully.")
        except Exception:
            print(f"  [NOT FOUND / SINGLE CLASS] Key '{kw}' skipped.")
            
    # Build metadata dataframe
    df_meta = pd.DataFrame({"Glycemia_Group": y})
    for kw, mapping in covariates.items():
        df_meta[kw] = df_meta.index.map(mapping)
        
    # 3. Preprocess & Aggregate Expression Data
    X_log2 = clean_expression_matrix(X_raw, control_prefix="AFFX-", standardise=False)
    X_genes = aggregate_enst_to_gene(X_log2, biomart_path)
    X_std = standardise_per_sample(X_genes, step_label="GSE18732 Standardisation")
    
    # 4. Fit Full PCA & Scree Plot Check
    print("\n--- SCREE ANALYSIS (First 20 PCs) ---")
    X_samples = X_std.T.values
    pca_full = PCA(n_components=20, random_state=42).fit(X_samples)
    explained_var = (pca_full.explained_variance_ratio_ * 100).round(2)
    print("Explained Variance Ratio per PC (%):")
    print(explained_var)
    
    # 5. Get 2D Coordinates & Identify Outlier
    pca2 = PCA(n_components=2, random_state=42)
    coords = pca2.fit_transform(X_samples)
    df_meta["PC1"] = coords[:, 0]
    df_meta["PC2"] = coords[:, 1]
    
    # Identify PC2 Outlier (< -30)
    outliers_pc2 = df_meta[df_meta["PC2"] < -30]
    print("\n--- PC2 EXTREME OUTLIER (< -30) ---")
    print(outliers_pc2[["PC1", "PC2", "Glycemia_Group"]])
    
    # Identify PC1 Tail Samples (> 10)
    tail_pc1 = df_meta[df_meta["PC1"] > 10]
    print(f"\n--- PC1 TAIL SAMPLES (> 10): {len(tail_pc1)} samples ---")
    print(tail_pc1.head(10))
    
    # 6. Plot Multi-Panel Covariate Exploration
    figures_dir = "project-a-convergence-classifier/figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create 2x2 grid for available covariates
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Subplot 1: Glycemia Group (Baseline)
    sns.scatterplot(
        data=df_meta, x="PC1", y="PC2", hue="Glycemia_Group",
        palette={"NGT": "#2ca02c", "IGT": "#ff7f0e", "DM": "#d62728"},
        s=80, alpha=0.85, ax=axes[0, 0]
    )
    axes[0, 0].set_title("Glycemia Group (NGT / IGT / DM)", fontweight="bold")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: Sex / Gender
    if "sex" in df_meta.columns or "gender" in df_meta.columns:
        col = "sex" if "sex" in df_meta.columns else "gender"
        sns.scatterplot(data=df_meta, x="PC1", y="PC2", hue=col, s=80, alpha=0.85, ax=axes[0, 1], palette="Set1")
        axes[0, 1].set_title(f"Coloured by {col.upper()}", fontweight="bold")
    else:
        axes[0, 1].text(0.5, 0.5, "Sex Metadata Not Extracted", ha="center")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Age or BMI
    num_col = None
    for c in ["age", "bmi"]:
        if c in df_meta.columns:
            num_col = c
            break
            
    if num_col:
        # Convert to numeric if possible
        df_meta[num_col] = pd.to_numeric(df_meta[num_col], errors="coerce")
        sns.scatterplot(data=df_meta, x="PC1", y="PC2", hue=num_col, palette="viridis", s=80, alpha=0.85, ax=axes[1, 0])
        axes[1, 0].set_title(f"Coloured by Continuous {num_col.upper()}", fontweight="bold")
    else:
        axes[1, 0].text(0.5, 0.5, "Age/BMI Metadata Not Extracted", ha="center")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: Scree Plot
    axes[1, 1].plot(range(1, 21), explained_var, marker="o", color="#1f77b4", linewidth=2)
    axes[1, 1].set_title("Scree Plot (First 20 Principal Components)", fontweight="bold")
    axes[1, 1].set_xlabel("Principal Component")
    axes[1, 1].set_ylabel("Explained Variance (%)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    out_fig = os.path.join(figures_dir, "gse18732_covariate_investigation.png")
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    print(f"\n[SUCCESS] Covariate investigation figure saved to: {out_fig}")
    plt.show()

if __name__ == "__main__":
    investigate_pc1_tail_and_outliers()