import pandas as pd
from load_geo import parse_metadata, load_dataset

data_path = "project-a-convergence-classifier/data/GSE18732_series_matrix.txt"

print("=== CHECKING TECHNICAL QC & MUSCLE FITNESS COVARIATES ===")

# Keywords relevant to RNA quality, batch, and skeletal muscle physiology
target_keywords = [
    "rin", "rna_integrity", "batch", "date", "scan",
    "vo2", "vo2_kg", "vo2max", "clamp", "glucose_infusion"
]

found_covariates = {}

for kw in target_keywords:
    try:
        cov_map = parse_metadata(data_path, target_keyword=kw)
        unique_vals = set(cov_map.values())
        print(f"  [FOUND] '{kw}': {len(unique_vals)} unique values -> Sample: {list(unique_vals)[:3]}")
        found_covariates[kw] = cov_map
    except Exception as e:
        print(f"  [NOT FOUND / SINGLE CLASS] '{kw}' skipped.")

# Print raw characteristic headers to verify if these variables exist under different names
print("\n=== RAW CHARACTERISTICS HEADERS SAMPLE ===")
with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("!Sample_characteristics_ch1") or line.startswith("!Sample_supplementary_file"):
            print(line.strip()[:120])