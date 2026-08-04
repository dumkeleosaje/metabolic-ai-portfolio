import numpy as np
import pandas as pd
from load_geo import load_dataset


def drop_affx_controls(df_matrix):
    # Drop Affymetrix technical control probes starting with 'AFFX-'
    # df_matrix is the raw gene expression matrix containing genes associated with patients
    # returns data frame containing only human transcript probes

    rows_before = df_matrix.shape[0]
    df_filtered = df_matrix[~df_matrix.index.str.startswith("AFFX-")].copy()
    rows_after = df_filtered.shape[0]

    print(f"[STEP 1] AFFX Filter: Rows before = {rows_before} | Rows after = {rows_after}")
    print(f"         Matrix shape after Step 1: {df_filtered.shape}")

    assert rows_after == 25710, f"Expected 25,710 rows remaining, but got {rows_after}"
    return df_filtered


def apply_log2_transform(df_matrix):
    df_log2 = np.log2(df_matrix + 1.0)
    # Fix: Use .values.max() to get scalar maximum across full matrix
    max_val = df_log2.values.max()

    print(f"[STEP 2] Log2 Transform: Max value after log2(x+1) = {max_val:.2f}")
    print(f"         Matrix shape after Step 2: {df_log2.shape}")

    # Explicit validation assertion 
    assert max_val < 18.0 and max_val > 17.0, f"Log2 max value validation failed: {max_val:.2f}"
    return df_log2


def standardize_per_sample(df_matrix):
    # Perform Z-score standardisation per sample (column-wise)
    sample_mean_before = df_matrix.mean(axis=0)
    sample_std_before = df_matrix.std(axis=0)

    # Calculate standardized values
    df_std = (df_matrix - sample_mean_before) / sample_std_before

    # Calculate post-standardization metrics for verification
    sample_mean_after = df_std.mean(axis=0)
    sample_std_after = df_std.std(axis=0)

    print(f"[STEP 3] Per-Sample Standardization Complete.")
    print(f"         Matrix shape after Step 3: {df_std.shape}")

    # TEST: Check if all sample means are 0 and stds are 1
    means_valid = np.allclose(sample_mean_after, 0.0, atol=1e-5)
    stds_valid = np.allclose(sample_std_after, 1.0, atol=1e-5)

    if not (means_valid and stds_valid):
        raise ValueError("CRITICAL ERROR: Per-sample standardization failed! Means or stds do not match expected Z-score bounds.")

    print("         [TEST PASSED] Verified: All per-sample means ≈ 0 and stds ≈ 1.")
    return df_std


def clean_expression_matrix(df_matrix):
    """Master pipeline executing AFFX removal, log2 transform, and per-sample standardization."""
    print("=== STARTING DATA CLEANING PIPELINE ===")
    df_step1 = drop_affx_controls(df_matrix)
    df_step2 = apply_log2_transform(df_step1)
    df_step3 = standardize_per_sample(df_step2)
    print("=== DATA CLEANING PIPELINE COMPLETE ===\n")
    return df_step3


if __name__ == "__main__":
    test_file_path = "project-a-convergence-classifier/data/GSE18732_series_matrix.txt"

    print("Loading raw dataset for cleaning verification...")
    X_raw, y = load_dataset(test_file_path)

    # Run cleaning pipeline
    X_clean = clean_expression_matrix(X_raw)

    print("=== SUMMARY OF CLEANED DATA ===")
    print(f"Cleaned Feature Matrix (X_clean) Shape: {X_clean.shape}")
    print(f"Total Missing Values (NaN):             {X_clean.isnull().sum().sum()}")