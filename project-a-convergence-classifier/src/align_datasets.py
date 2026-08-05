import pandas as pd
import numpy as np
from load_geo import load_dataset
from clean_geo import clean_expression_matrix, standardise_per_sample
from aggregate_genes import aggregate_enst_to_gene


def aggregate_gpl_to_gene(df_matrix, gpl_file_path, probe_col="ID"):
    """Map array probes to HGNC gene symbols with strict column prioritization."""
    initial_probe_count = df_matrix.shape[0]

    # Find where the actual table header starts (skip metadata lines starting with ^ or !)
    header_line = 0
    with open(gpl_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if line.startswith("ID") or line.startswith('"ID"') or ("\t" in line and not line.startswith("^") and not line.startswith("!")):
                header_line = i
                break

    gpl_df = pd.read_csv(gpl_file_path, sep="\t", skiprows=header_line, low_memory=False)
    gpl_df.columns = gpl_df.columns.str.strip('" ')

    p_col = probe_col if probe_col in gpl_df.columns else gpl_df.columns[0]

    # Strict column order: Prioritize official HGNC gene symbol columns over descriptive titles
    preferred_symbols = ["Gene Symbol", "Gene_Symbol", "Symbol", "ILMN_Gene", "Gene_symbol"]
    symbol_col = None

    for col in preferred_symbols:
        if col in gpl_df.columns:
            symbol_col = col
            break

    if not symbol_col:
        # Secondary fallback: Match exact symbol string, avoiding 'title' or 'description'
        for col in gpl_df.columns:
            if "symbol" in col.lower() and "title" not in col.lower() and "desc" not in col.lower():
                symbol_col = col
                break

    if not symbol_col:
        raise ValueError(f"Could not locate a valid HGNC Gene Symbol column in {gpl_file_path}. Available columns: {gpl_df.columns.tolist()[:10]}")

    print(f"  [GPL DETECTED] Using probe col: '{p_col}', symbol col: '{symbol_col}'")

    # Clean missing values and multiple gene mappings (e.g., 'GENE1 /// GENE2')
    gpl_df = gpl_df.dropna(subset=[symbol_col]).copy()
    gpl_df[symbol_col] = gpl_df[symbol_col].astype(str).str.split("///").str[0].str.strip(' "')
    gpl_df = gpl_df[gpl_df[symbol_col] != ""]

    map_dict = dict(zip(gpl_df[p_col].astype(str).str.strip(' "'), gpl_df[symbol_col]))

    df_work = df_matrix.copy()
    df_work.index = df_work.index.astype(str).str.strip(' "')
    df_work["gene_symbol"] = df_work.index.map(map_dict)

    mapped_probes = df_work["gene_symbol"].notnull().sum()
    df_mapped = df_work.dropna(subset=["gene_symbol"]).copy()

    # Average duplicate probe intensities per unique gene symbol
    df_gene_level = df_mapped.groupby("gene_symbol").mean(numeric_only=True)

    print(f"  [GPL SUMMARY] {mapped_probes}/{initial_probe_count} probes mapped ({mapped_probes/initial_probe_count*100:.1f}%) → {df_gene_level.shape[0]} unique HGNC genes.")
    return df_gene_level


def align_three_datasets(paths_dict, biomart_path, gpl570_path, gpl14951_path):
    """Load 3 GEO datasets, aggregate to HGNC symbols, intersect feature spaces, and standardise."""
    
    print("=== PHASE 1: INDIVIDUAL PREPROCESSING & AGGREGATION ===")

    # 1. GSE18732 (Muscle - ENST Custom CDF)
    print("\n--- [1/3] Processing GSE18732 (Muscle - Affymetrix ENST) ---")
    X1_raw, y1 = load_dataset(paths_dict["GSE18732"], target_keyword="glycemiagroup")
    X1_log2 = clean_expression_matrix(X1_raw, control_prefix="AFFX-", standardise=False)
    X1_genes = aggregate_enst_to_gene(X1_log2, biomart_path)

    # 2. GSE10946 (PCOS - Affymetrix GPL570)
    print("\n--- [2/3] Processing GSE10946 (PCOS - Affymetrix GPL570) ---")
    X2_raw, y2 = load_dataset(paths_dict["GSE10946"], target_keyword="pcos")
    X2_log2 = clean_expression_matrix(X2_raw, control_prefix="AFFX-", standardise=False)
    X2_genes = aggregate_gpl_to_gene(X2_log2, gpl570_path, probe_col="ID")

    # 3. GSE89632 (NAFLD/Liver - Illumina GPL14951)
    print("\n--- [3/3] Processing GSE89632 (Liver - Illumina GPL14951) ---")
    X3_raw, y3 = load_dataset(paths_dict["GSE89632"], target_keyword="diagnosis")
    X3_log2 = clean_expression_matrix(X3_raw, control_prefix=None, standardise=False)
    X3_genes = aggregate_gpl_to_gene(X3_log2, gpl14951_path, probe_col="ID")

    print("\n=== PHASE 2: PLATFORM OVERLAP & 3-WAY INTERSECTION ===")
    genes1 = set(X1_genes.index)
    genes2 = set(X2_genes.index)
    genes3 = set(X3_genes.index)

    overlap_12 = len(genes1 & genes2)
    overlap_13 = len(genes1 & genes3)
    overlap_23 = len(genes2 & genes3)

    print(f"  Pairwise Overlap GSE18732 & GSE10946 (Affy vs Affy):    {overlap_12} genes")
    print(f"  Pairwise Overlap GSE18732 & GSE89632 (Affy vs Illumina): {overlap_13} genes")
    print(f"  Pairwise Overlap GSE10946 & GSE89632 (Affy vs Illumina): {overlap_23} genes")

    # Compute shared 3-way gene intersection
    shared_genes = sorted(list(genes1 & genes2 & genes3))
    print(f"\n[FINAL 3-WAY INTERSECTION FEATURE SPACE]: {len(shared_genes)} genes")

    if len(shared_genes) < 5000:
        raise ValueError("CRITICAL WARNING: 3-way gene intersection dropped below 5,000 genes! Check annotation mapping tables.")

    print("\n=== PHASE 3: SUBSETTING & PER-DATASET STANDARDISATION ===")
    aligned_matrices = {}
    labels_dict = {"GSE18732": y1, "GSE10946": y2, "GSE89632": y3}
    unaligned = {"GSE18732": X1_genes, "GSE10946": X2_genes, "GSE89632": X3_genes}

    for name, df in unaligned.items():
        # Subset rows strictly to the 3-way gene intersection
        df_subset = df.loc[shared_genes].copy()
        
        # Standardise per-sample ON THE ALIGNED FEATURE SPACE
        df_std = standardise_per_sample(df_subset, step_label=f"{name} Standardisation")
        aligned_matrices[name] = df_std
        print(f"  {name} final shape: {df_std.shape} (Genes x Patients)")

    return aligned_matrices, labels_dict, shared_genes


if __name__ == "__main__":
    paths = {
        "GSE18732": "project-a-convergence-classifier/data/GSE18732_series_matrix.txt",
        "GSE10946": "project-a-convergence-classifier/data/GSE10946_series_matrix.txt",
        "GSE89632": "project-a-convergence-classifier/data/GSE89632_series_matrix.txt"
    }
    biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
    gpl570 = "project-a-convergence-classifier/data/GPL570.annot"
    gpl14951 = "project-a-convergence-classifier/data/GPL14951.annot"

    matrices, labels, gene_intersection = align_three_datasets(paths, biomart_path, gpl570, gpl14951)