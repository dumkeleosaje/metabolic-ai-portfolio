from load_geo import load_dataset, parse_metadata

def verify_target_labels():
    path_10946 = "project-a-convergence-classifier/data/GSE10946_series_matrix.txt"
    path_89632 = "project-a-convergence-classifier/data/GSE89632_series_matrix.txt"

    print("=== INSPECTING RAW METADATA LINES FOR GSE10946 ===")
    with open(path_10946, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("!Sample_characteristics_ch1") or line.startswith("!Sample_title"):
                print(f"Line {i}: {line.strip()[:110]}")

    print("\n=== CHECKING GSE89632 LABEL DISTRIBUTION ===")
    _, y3 = load_dataset(path_89632, target_keyword="diagnosis")
    print(y3.value_counts())

if __name__ == "__main__":
    verify_target_labels()