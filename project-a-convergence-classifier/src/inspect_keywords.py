def inspect_metadata_headers(file_path, dataset_name):
    print(f"\n=== METADATA HEADERS FOR {dataset_name} ===")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("!Sample_characteristics_ch1"):
                # Print unique key prefix (e.g., "!Sample_characteristics_ch1: disease status: PCOS")
                print("  ", line.strip()[:100])

if __name__ == "__main__":
    inspect_metadata_headers("project-a-convergence-classifier/data/GSE18732_series_matrix.txt", "GSE18732")
    inspect_metadata_headers("project-a-convergence-classifier/data/GSE10946_series_matrix.txt", "GSE10946")
    inspect_metadata_headers("project-a-convergence-classifier/data/GSE89632_series_matrix.txt", "GSE89632")