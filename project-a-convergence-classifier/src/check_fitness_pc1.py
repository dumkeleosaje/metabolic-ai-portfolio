import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from load_geo import load_dataset
from clean_geo import clean_expression_matrix, standardise_per_sample
from aggregate_genes import aggregate_enst_to_gene

def parse_gse18732_fitness(file_path):
    #Extract vo2_kg values directly from GSE18732 metadata lines
    sample_ids = []
    vo2_values = []
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [item.strip().strip('"') for item in line.split("\t")[1:]]
            # Match exact vo2_kg key without matching log_vo2_kg or vo2_kg_ffm
            if line.startswith("!Sample_characteristics_ch1") and '"vo2_kg:' in line:
                raw_items = line.split("\t")[1:]
                for item in raw_items:
                    val_str = item.strip().strip('"').replace("vo2_kg:", "").strip()
                    try:
                        vo2_values.append(float(val_str))
                    except ValueError:
                        vo2_values.append(np.nan)
                break  # Stop after processing the primary vo2_kg line
                        
    return pd.Series(vo2_values, index=sample_ids, name="vo2_kg")

def evaluate_vo2_pc1():
    data_path = "project-a-convergence-classifier/data/GSE18732_series_matrix.txt"
    biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
    
    X_raw, y = load_dataset(data_path, target_keyword="glycemiagroup")
    s_vo2 = parse_gse18732_fitness(data_path)
    
    # Preprocess & Aggregate
    X_log2 = clean_expression_matrix(X_raw, control_prefix="AFFX-", standardise=False)
    X_genes = aggregate_enst_to_gene(X_log2, biomart_path)
    X_std = standardise_per_sample(X_genes, step_label="GSE18732")
    
    # Fit PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_std.T.values)
    
    df_res = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "vo2_kg": s_vo2,
        "Glycemia_Group": y
    })
    
    # Check Pearson correlation between PC1 and VO2_max
    df_clean = df_res.dropna(subset=["vo2_kg"])
    corr = df_clean["PC1"].corr(df_clean["vo2_kg"])
    print(f"\n=== PEARSON CORRELATION (PC1 vs VO2_kg): r = {corr:.3f} ===")
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_res, x="PC1", y="PC2", hue="vo2_kg", palette="magma", s=90)
    plt.title(f"GSE18732 PCA Coloured by VO2_kg (Fitness)\nCorrelation with PC1: r = {corr:.3f}", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("project-a-convergence-classifier/figures/gse18732_vo2_pca.png", dpi=300, bbox_inches="tight")
    print("Saved figure to figures/gse18732_vo2_pca.png")
    plt.show()

if __name__ == "__main__":
    evaluate_vo2_pc1()
