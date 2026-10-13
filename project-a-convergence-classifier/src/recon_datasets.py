from load_geo import load_dataset

datasets = {
    "GSE18732": "project-a-convergence-classifier/data/GSE18732_series_matrix.txt",
    "GSE10946": "project-a-convergence-classifier/data/GSE10946_series_matrix.txt",
    "GSE89632": "project-a-convergence-classifier/data/GSE89632_series_matrix.txt"
}

print("=== GEO DATASET RECONNAISSANCE ===")
for name, path in datasets.items():
    print(f"\n--- {name} ---")
    try:
        # Load matrix with generic target keyword to inspect metadata
        X, y = load_dataset(path, target_keyword="characteristics_ch1")
        print(f"Raw Expression Shape (Probes x Samples): {X.shape}")
        print(f"Expression Max Value: {X.values.max():.2f}")
        print(f"Sample Probe IDs (Index head): {X.index[:3].tolist()}")
        print(f"Unique Phenotype Categories Found: {set(y.values[:5])}")
    except Exception as e:
        print(f"Error inspecting {name}: {e}")