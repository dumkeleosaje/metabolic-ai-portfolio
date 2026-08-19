import pytest
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from load_simulation import generate_in_silico_cgm
from extract_features import compute_glucose_features
from fit_hmm import fit_metabolic_hmm
from evaluate_states import compute_state_transition_matrix, compute_dwell_times_with_markov_surrogate, analyze_recovery_dynamics


def test_in_silico_cgm_generation(tmp_path):
    """Verifies in silico time-series generation, stationarity, and schema validity."""
    test_csv = os.path.join(tmp_path, "test_patient.csv")
    df = generate_in_silico_cgm(test_csv, days=2, seed=42)
    
    assert os.path.exists(test_csv)
    assert len(df) == 2 * 288
    assert "true_state" in df.columns
    assert "glucose" in df.columns
    assert "insulin_on_board" in df.columns
    assert set(df["true_state"].unique()).issubset({"Fasting_Basal", "Absorption_Spike", "Insulin_Clearance"})


def test_feature_engineering():
    """Verifies calculation of rate of change, rolling variability, and clean NaNs."""
    times = pd.date_range("2025-01-01", periods=30, freq="5min")
    glucose_vals = np.linspace(100, 160, 30)
    df_sample = pd.DataFrame({
        "timestamp": times,
        "glucose": glucose_vals,
        "true_state": ["Fasting_Basal"] * 30,
        "insulin_on_board": [0.5] * 30
    })
    
    df_feats = compute_glucose_features(df_sample)
    
    assert "dG_dt" in df_feats.columns
    assert "rolling_std_1h" in df_feats.columns
    assert "glucose_smooth" in df_feats.columns
    assert not df_feats.isnull().any().any()


def test_hmm_fitting_and_recovery_dynamics():
    """Verifies HMM training, state decoding, and recovery metrics."""
    times = pd.date_range("2025-01-01", periods=150, freq="5min")
    g_stable = np.full(50, 100.0)
    g_spike = np.linspace(100, 170, 50)
    g_decay = np.linspace(170, 100, 50)
    glucose = np.concatenate([g_stable, g_spike, g_decay])
    
    df_synth = pd.DataFrame({
        "timestamp": times,
        "glucose": glucose,
        "true_state": ["Fasting_Basal"] * 50 + ["Absorption_Spike"] * 50 + ["Insulin_Clearance"] * 50,
        "insulin_on_board": [0.0] * 50 + [1.5] * 50 + [0.8] * 50
    })
    df_feats = compute_glucose_features(df_synth)
    
    model, df_decoded, profiles = fit_metabolic_hmm(df_feats, n_states=3, random_state=42, save_artifacts=False)
    
    assert "raw_state" in df_decoded.columns
    assert "metabolic_state" in df_decoded.columns
    assert len(np.unique(df_decoded["raw_state"])) == 3
    
    trans_mat, _ = compute_state_transition_matrix(df_decoded)
    assert np.allclose(trans_mat.sum(axis=1), 1.0)
    
    rec_metrics = analyze_recovery_dynamics(df_decoded)
    assert "High-Velocity Excursion Events Detected" in rec_metrics
    assert "Resolution Completion Rate %" in rec_metrics