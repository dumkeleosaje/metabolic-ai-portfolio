import pandas as pd
from load_geo import load_dataset
from align_datasets import align_three_datasets

path_10946 = "project-a-convergence-classifier/data/GSE10946_series_matrix.txt"
path_18732 = "project-a-convergence-classifier/data/GSE18732_series_matrix.txt"
path_89632 = "project-a-convergence-classifier/data/GSE89632_series_matrix.txt"

print("=== CHECK 1: GSE10946 TARGET LABEL DISTRIBUTION ===")
_, y2 = load_dataset(path_10946, target_keyword="pcos")
print(y2.value_counts())

print("\n=== CHECK 2: JUNK / NON-STANDARD SYMBOLS IN 10,369 INTERSECTION ===")
biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
gpl570 = "project-a-convergence-classifier/data/GPL570.annot"
gpl14951 = "project-a-convergence-classifier/data/GPL14951.annot"

paths = {"GSE18732": path_18732, "GSE10946": path_10946, "GSE89632": path_89632}
matrices, labels, gene_intersection = align_three_datasets(paths, biomart_path, gpl570, gpl14951)

junk_symbols = [g for g in gene_intersection if g.startswith(("LOC", "LINC", "hCG_", "MIR", "ORF", "C1or", "C2or", "C3or"))]
print(f"Total Non-Standard / Locus Tags Found in Intersection: {len(junk_symbols)} / {len(gene_intersection)}")
print("Sample Non-Standard Symbols:", junk_symbols[:20])
