import pandas as pd
import numpy as np 
from load_geo import load_dataset
from clean_geo import clean_expression_matrix, standardise_per_sample

def aggregate_enst_to_gene (df_matrix, mapping_file_path):
    #Map microarray probes (ENST) to HGNC gene symbols 

    initial_probe_count = df_matrix.shape[0]

    #Copy dataframe and strip '_at' suffix from row strings
    df_work = df_matrix.copy()
    df_work.index= df_work.index.str.replace(r"_at$", "", regex=True)

    #Load the BioMart mapping file into a data frame
    mapping_df = pd.read_csv(mapping_file_path, sep="\t")

    #Drop rows where HGNC symbol is missing 
    mapping_df = mapping_df.dropna(subset=["HGNC symbol"]).copy()
    mapping_df["HGNC symbol"] = mapping_df["HGNC symbol"].str.strip()
    mapping_df = mapping_df[mapping_df["HGNC symbol"] != ""]

    #Create a dictionary mapping transcript IDs to gene symbol 
    map_dict = dict(zip(mapping_df["Transcript stable ID"], mapping_df["HGNC symbol"]))

    df_work["gene_symbol"] = df_work.index.map(map_dict)

    mapped_probes = df_work["gene_symbol"].notnull().sum()
    unmapped_probes = df_work["gene_symbol"].isnull().sum()

    print(f"[AGGREGATION SUMMARY]")
    print(f"  Initial Probes:     {initial_probe_count}")
    print(f"  Mapped to Genes:    {mapped_probes} ({mapped_probes / initial_probe_count * 100:.1f}%)")
    print(f"  Unmapped / Dropped: {unmapped_probes} ({unmapped_probes / initial_probe_count * 100:.1f}%)")

    # LINE 7: Drop probes that could not be mapped to any HGNC gene symbol
    df_mapped = df_work.dropna(subset=["gene_symbol"]).copy()

    # LINE 8: Group rows by gene_symbol and compute mean across duplicate transcripts (numeric_only=True)
    df_gene_level = df_mapped.groupby("gene_symbol").mean(numeric_only=True)

    print(f"  Final Unique Genes: {df_gene_level.shape[0]}")
    return df_gene_level

if __name__ == "__main__":
    test_matrix_path = "project-a-convergence-classifier/data/GSE18732_series_matrix.txt"
    mapping_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"

    print("Running end-to-end Load -> Clean -> Aggregate -> Standardise pipeline...\n")
    # Load raw data
    X_raw, y = load_dataset(test_matrix_path, target_keyword="glycemiagroup")

    # Clean data (AFFX filter + log2, defer standardisation to post-aggregation)
    X_log2 = clean_expression_matrix(X_raw, control_prefix="AFFX-", standardise=False)

    # Aggregate transcript probes to HGNC gene symbols
    X_genes_unscaled = aggregate_enst_to_gene(X_log2, mapping_path)

    # Standardise per-sample AFTER aggregation on the final gene feature space
    print("\n[STEP 3] Standardising aggregated gene matrix per sample...")
    X_genes = standardise_per_sample(X_genes_unscaled, step_label="STEP 3")

    print("\n=== FINAL AGGREGATED FEATURE MATRIX ===")
    print(f"Shape (Genes x Patients): {X_genes.shape}")
    print(f"Missing Values (NaN):     {X_genes.isnull().sum().sum()}")
    print("\nVerifying Per-Sample Means & Standard Deviations:")
    print(f"  Sample Means Summary:  min={X_genes.mean(axis=0).min():.6f}, max={X_genes.mean(axis=0).max():.6f}")
    print(f"  Sample Stds Summary:   min={X_genes.std(axis=0).min():.6f}, max={X_genes.std(axis=0).max():.6f}")