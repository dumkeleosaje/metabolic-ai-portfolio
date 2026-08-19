import os
import numpy as np
import pandas as pd
from load_simulation import generate_in_silico_cgm


def compute_glucose_features(df_sim):
    """
    Computes smoothed physiological features for HMM state segmentation:
      1. glucose: Raw sensor reading (mg/dL)
      2. glucose_smooth: 30-minute rolling average (suppresses high-frequency jitter)
      3. dG_dt: 15-minute rate of change (mg/dL/min)
      4. rolling_std_1h: 1-hour rolling glycemic volatility
    """
    df = df_sim.copy().sort_values("timestamp").reset_index(drop=True)
    
    # 1. Smoothed baseline
    df["glucose_smooth"] = df["glucose"].rolling(window=6, min_periods=6, center=False).mean()
    
    # 2. Rate of change over 15 minutes (3 steps)
    df["dG_dt"] = (df["glucose_smooth"] - df["glucose_smooth"].shift(3)) / 15.0
    
    # 3. Rolling Glycemic Volatility (1-hour window = 12 steps)
    df["rolling_std_1h"] = df["glucose"].rolling(window=12, min_periods=12).std()
    
    # Drop initialization warmup window (12 steps = 1 hour) instead of fabricating values
    df = df.dropna().reset_index(drop=True)
    return df


if __name__ == "__main__":
    sim_path = "project-c-state-modeller/data/simulated_patient_A.csv"
    if not os.path.exists(sim_path):
        generate_in_silico_cgm(sim_path, days=14)
        
    df_sim = pd.read_csv(sim_path)
    df_features = compute_glucose_features(df_sim)
    
    output_csv = "project-c-state-modeller/data/cgm_engineered_features.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_features.to_csv(output_csv, index=False)
    
    print("\n==================================================")
    print("=== CGM PHYSIOLOGICAL FEATURE EXTRACTION ===")
    print("==================================================")
    print(f"Total Valid Samples: {len(df_features)} (Warmup dropped)")
    print(df_features[["glucose", "glucose_smooth", "dG_dt", "rolling_std_1h"]].describe().round(3))