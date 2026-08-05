import numpy as np
import pandas as pd
from load_geo import load_dataset


def drop_control_probes(df_matrix, control_prefix ="AFFX-"):

    if not control_prefix:
        print("[STEP 1] Control Probe Filter: Skipped (No prefix specified).")
        return df_matrix.copy()

    rows_before = df_matrix.shape[0]
    df_filtered = df_matrix[~df_matrix.index.str.startswith(control_prefix)].copy()
    rows_after = df_filtered.shape[0]
    dropped_count = rows_before - rows_after

    print(f"[STEP 1] Control Probe Filter ('{control_prefix}'):")
    print(f"         Removed {dropped_count} control probes ({rows_before} → {rows_after} rows remaining).")
    return df_filtered


def apply_log2_transform(df_matrix, prelog_threshold=20.0):
    #Applies transformation if data not already log-transformed
    #Checks if max_value is less than threshold ans skips transformation to avoid double logging

    input_max = df_matrix.values.max()

    if input_max <= prelog_threshold:
        print(f"[STEP 2] Log2 Transform: SKIPPED. Input max ({input_max:.2f}) <= {prelog_threshold}.")
        print("         Data is already log-transformed.")
        return df_matrix.copy()

    df_log2 = np.log2(df_matrix + 1.0)
    output_max = df_log2.values.max()

    print(f"[STEP 2] Log2 Transform: APPLIED. Input max ({input_max:.2f}) → Output max ({output_max:.2f}).")
    return df_log2


def standardise_per_sample(df_matrix, step_label="STEP 3"):
    #Perform per-sample standardisation to remove chip-level bias 

    # Column-wise mean and std (pandas ddof=1)
    sample_mean_before = df_matrix.mean(axis=0)
    sample_std_before = df_matrix.std(axis=0)

    df_std = (df_matrix - sample_mean_before) / sample_std_before

    # Calculate post-standardization metrics for verification
    sample_mean_after = df_std.mean(axis=0)
    sample_std_after = df_std.std(axis=0)

    print(f"[{step_label}] Per-Sample Standardization Complete. Shape: {df_std.shape}")

    # TEST: Dataset-agnostic validation (all per-sample means ≈ 0 and stds ≈ 1)
    means_valid = np.allclose(sample_mean_after, 0.0, atol=1e-5)
    stds_valid = np.allclose(sample_std_after, 1.0, atol=1e-5)

    if not (means_valid and stds_valid):
        raise ValueError("CRITICAL ERROR: Per-sample standardization failed! Means or stds do not match expected Z-score bounds.")

    print("         [TEST PASSED] Verified: All per-sample means ≈ 0 and stds ≈ 1.")
    return df_std


def clean_expression_matrix(df_matrix, control_prefix="AFFX-", prelog_threshold=20.0, standardise=False):
    #reusable code for control probe filtering, adaptive log2 transform, and per-sample standardization."""
    print("=== STARTING DATA CLEANING PIPELINE ===")
    df_step1 = drop_control_probes(df_matrix, control_prefix=control_prefix)
    df_step2 = apply_log2_transform(df_step1, prelog_threshold=prelog_threshold)

    if standardise:
        df_out = standardise_per_sample(df_step2, step_label="STEP 3")
    else:
        df_out = df_step2
    print("=== DATA CLEANING PIPELINE COMPLETE ===\n")
    return df_out


if __name__ == "__main__":
    test_file_path = "project-a-convergence-classifier/data/GSE18732_series_matrix.txt"

    print("Testing clean_geo standalone...")
    X_raw, y = load_dataset(test_file_path)
    X_clean = clean_expression_matrix(X_raw, control_prefix="AFFX-", standardise=True)