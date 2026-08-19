import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_in_silico_cgm(output_path="project-c-state-modeller/data/simulated_patient_A.csv", days=14, seed=42):
    """
    Generates a stationary in silico physiological CGM time-series with 3 balanced ground-truth states:
      - Fasting_Basal (0): Overnight / inter-meal steady state baseline (80-110 mg/dL)
      - Absorption_Spike (1): Active carbohydrate absorption driving postprandial rise (130-220 mg/dL)
      - Insulin_Clearance (2): Active post-peak insulin action mediating downward recovery
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(seed)
    
    start_time = datetime(2025, 5, 1, 0, 0, 0)
    total_steps = days * 288  # 288 5-minute intervals per day
    
    current_glucose = 95.0
    gut_carbs = 0.0
    active_insulin = 0.0
    
    records = []
    
    for step in range(total_steps):
        t = start_time + timedelta(minutes=5 * step)
        hour = t.hour + t.minute / 60.0
        
        # 1. Meals (Breakfast 08:00, Lunch 13:00, Dinner 19:30)
        is_meal = False
        carbs = 0.0
        if t.minute == 0 and t.hour == 8 and np.random.rand() > 0.05:
            carbs = float(np.random.choice([45, 60, 75]))
            is_meal = True
        elif t.minute == 0 and t.hour == 13 and np.random.rand() > 0.05:
            carbs = float(np.random.choice([55, 75, 90]))
            is_meal = True
        elif t.minute == 30 and t.hour == 19 and np.random.rand() > 0.05:
            carbs = float(np.random.choice([65, 85, 105]))
            is_meal = True
            
        bolus_dose = 0.0
        if is_meal:
            gut_carbs += carbs
            # 8% probability of an under-bolus for realistic postprandial glycemic excursions
            bolus_ratio = 0.80 if np.random.rand() < 0.08 else 1.0
            bolus_dose = round((carbs / 10.0) * bolus_ratio + np.random.normal(0, 0.15), 1)
            active_insulin += max(0.5, bolus_dose)
                
        # 2. Two-Compartment Pharmacokinetics
        carb_absorption = gut_carbs * 0.060
        gut_carbs -= carb_absorption
        
        insulin_action = active_insulin * 0.035
        active_insulin -= insulin_action
        
        # 3. Circadian baseline + balanced homeostasis pull
        circadian_drift = 5.0 * np.sin(2 * np.pi * (hour - 4) / 24.0)
        homeostasis_pull = 0.020 * (95.0 + circadian_drift - current_glucose)
        
        # True physiological delta before sensor noise
        true_delta_g = (carb_absorption * 3.0) - (insulin_action * 20.0) + homeostasis_pull
        
        # 4. Balanced 3-Compartment Ground-Truth
        # Excursion: active gut carb absorption driving upward trend
        # Clearance: elevated active insulin mediating active downward disposal
        # Basal: quiet inter-meal/overnight periods
        if carb_absorption > 0.15 and true_delta_g > 0.02:
            ground_truth_state = "Absorption_Spike"
        elif active_insulin > 0.45 and true_delta_g < -0.01:
            ground_truth_state = "Insulin_Clearance"
        else:
            ground_truth_state = "Fasting_Basal"
            
        # Add sensor measurement noise
        sensor_noise = np.random.normal(0, 0.6)
        current_glucose += (true_delta_g + sensor_noise)
        current_glucose = float(np.clip(current_glucose, 50.0, 280.0))
        
        records.append({
            "timestamp": t,
            "glucose": round(current_glucose, 1),
            "carbs_ingested": carbs if is_meal else 0.0,
            "bolus_dose": bolus_dose,
            "insulin_on_board": round(active_insulin, 2),
            "true_state": ground_truth_state
        })

    df = pd.DataFrame(records)
    
    # Assertions
    max_g = df["glucose"].max()
    min_g = df["glucose"].min()
    assert max_g < 275.0, f"Error: Simulation clipped ({max_g} mg/dL)."
    assert min_g > 55.0, f"Error: Severe hypoglycemia ({min_g} mg/dL)."
    
    half_pt = len(df) // 2
    w1_mean = df.iloc[:half_pt]["glucose"].mean()
    w2_mean = df.iloc[half_pt:]["glucose"].mean()
    drift = abs(w1_mean - w2_mean)
    assert drift < 15.0, f"Error: Non-stationary simulation drift: Week 1 ({w1_mean:.1f}) vs Week 2 ({w2_mean:.1f})"
    
    df.to_csv(output_path, index=False)
    print(f"[In Silico Generator] Generated {len(df)} samples ({days} days) at: {output_path}")
    print(f"  Glucose Range: {min_g:.1f} - {max_g:.1f} mg/dL (Mean: {df['glucose'].mean():.1f} mg/dL)")
    print(f"  Stationarity: Week 1 Mean = {w1_mean:.1f} mg/dL | Week 2 Mean = {w2_mean:.1f} mg/dL (Drift: {drift:.2f} mg/dL)")
    return df


if __name__ == "__main__":
    df_sim = generate_in_silico_cgm()
    print("\n==================================================")
    print("=== IN SILICO PHYSIOLOGICAL GROUND TRUTH ===")
    print("==================================================")
    print("Ground-Truth State Proportions (%):")
    print(df_sim["true_state"].value_counts(normalize=True).round(3) * 100)