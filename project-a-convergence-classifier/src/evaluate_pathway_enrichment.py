import warnings
import pandas as pd
import numpy as np
import gseapy as gp
from align_datasets import align_three_datasets

warnings.filterwarnings("ignore")


def compute_lfc(X_raw, y_binary):
    ctrl = X_raw.loc[:, y_binary == 0].mean(axis=1)
    dis = X_raw.loc[:, y_binary == 1].mean(axis=1)
    return dis - ctrl


def run_pathway_enrichment():
    print("\n==================================================")
    print("=== MUSCLE-LIVER CONCORDANT PATHWAY ENRICHMENT ===")
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
    
    # Calculate Log-Fold Changes
    mask_mus = labels_raw["GSE18732"].isin(["NGT", "DM"])
    y_mus = (labels_raw["GSE18732"][mask_mus] == "DM").astype(int)
    lfc_mus = compute_lfc(matrices_raw["GSE18732"].loc[:, mask_mus], y_mus)
    
    mask_liv = labels_raw["GSE89632"].isin(["HC", "NASH"])
    y_liv = (labels_raw["GSE89632"][mask_liv] == "NASH").astype(int)
    lfc_liv = compute_lfc(matrices_raw["GSE89632"].loc[:, mask_liv], y_liv)
    
    # Concordance product: Positive value = shifted in the same direction in both tissues
    concordance = lfc_mus * lfc_liv
    top_concordant_genes = concordance.nlargest(200).index.tolist()
    
    print(f"Extracting Top 200 Concordant Genes between Muscle & Liver...")
    print("Sample Genes:", top_concordant_genes[:10])
    
    # Save top concordant genes to CSV
    pd.Series(top_concordant_genes, name="Gene_Symbol").to_csv(
        "project-a-convergence-classifier/data/muscle_liver_concordant_genes.csv", index=False
    )
    
    # Run Enrichr using 'Homo Sapiens'
    try:
        enr = gp.enrichr(
            gene_list=top_concordant_genes,
            gene_sets=['KEGG_2021_Human', 'MSigDB_Hallmark_2020'],
            organism='Homo Sapiens',
            outdir=None
        )
        print("\n=== TOP ENRICHED CONVERGENT PATHWAYS ===")
        df_res = enr.results[['Gene_set', 'Term', 'Adjusted P-value', 'Overlap', 'Genes']].head(10)
        print(df_res.to_string(index=False))
    except Exception as e:
        print(f"\n[INFO] Enrichr query skipped or offline: {e}")
        print("Saved top 200 concordant genes to data/muscle_liver_concordant_genes.csv.")


if __name__ == "__main__":
    run_pathway_enrichment()