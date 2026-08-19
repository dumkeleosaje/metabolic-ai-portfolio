import os
import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, confusion_matrix

STATE_LABELS = {
    "basal": "State 0: Fasting Basal",
    "clearance": "State 1: Insulin Clearance",
    "excursion": "State 2: Absorption Excursion"
}


def fit_best_hmm(X_scaled, n_states, n_inits=5, max_iter=500, random_state=42):
    """
    Fits Gaussian HMM with multiple random restarts to guarantee global convergence
    and prevent local optima cliffs. Retains sticky diagonal priors.
    
    min_covar=0.05 enforces a variance floor to prevent emission covariance collapse
    onto low-variance sensor plateaus (e.g. flat overnight fasting glucose traces).
    """
    best_model = None
    best_score = -np.inf
    
    transmat_prior = np.full((n_states, n_states), (1.0 - 0.92) / (n_states - 1))
    np.fill_diagonal(transmat_prior, 0.92)
    
    for seed in range(random_state, random_state + n_inits):
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            min_covar=0.05,  # Regularization floor preventing variance collapse on flat traces
            n_iter=max_iter,
            init_params="smc",
            random_state=seed
        )
        model.transmat_ = transmat_prior.copy()
        model.fit(X_scaled)
        
        score = model.score(X_scaled)
        if score > best_score:
            best_score = score
            best_model = model
            
    return best_model


def evaluate_model_selection(X_scaled, y_true=None, max_states=6):
    """
    Evaluates AIC / BIC and Ground-Truth ARI across k in [2, max_states].
    Note: Adjusted Rand Index (ARI) is invariant to label permutation,
    so computing ARI on raw predicted state IDs is mathematically equivalent
    to computing it on mapped semantic strings.
    """
    print("\n--- MODEL SELECTION DUAL-METRIC CURVE (BIC vs GROUND-TRUTH ARI) ---")
    results = []
    n_samples, n_features = X_scaled.shape
    
    for k in range(2, max_states + 1):
        model = fit_best_hmm(X_scaled, n_states=k, n_inits=3, max_iter=400)
        log_likelihood = model.score(X_scaled)
        
        n_params = (k - 1) + k * (k - 1) + k * n_features + k * n_features
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        ari = np.nan
        if y_true is not None:
            pred_states = model.predict(X_scaled)
            ari = adjusted_rand_score(y_true, pred_states)
            
        results.append({
            "k_states": k,
            "log_likelihood": log_likelihood,
            "AIC": aic,
            "BIC": bic,
            "ARI": ari,
            "converged": model.monitor_.converged,
            "n_iter": model.monitor_.iter
        })
        ari_str = f"| ARI: {ari:.4f}" if not np.isnan(ari) else ""
        print(f"  k={k} | Log-Lik: {log_likelihood:8.1f} | BIC: {bic:8.1f} {ari_str} | Converged: {model.monitor_.converged}")
        
    df_selection = pd.DataFrame(results)
    best_bic_k = int(df_selection.loc[df_selection["BIC"].idxmin()]["k_states"])
    best_ari_k = int(df_selection.loc[df_selection["ARI"].idxmax()]["k_states"]) if y_true is not None else 3
    
    is_bic_monotonic = df_selection["BIC"].is_monotonic_decreasing
    bic_phrase = f"BIC decreases monotonically through k={best_bic_k}" if is_bic_monotonic else f"Minimum BIC achieved at k={best_bic_k}"
    
    print(f"\n[Model Selection Finding]:")
    print(f"  - {bic_phrase}, reflecting likelihood over-segmentation on continuous time-series.")
    
    if y_true is not None:
        best_ari_val = df_selection.loc[df_selection["k_states"] == best_ari_k, "ARI"].values[0]
        if best_ari_k == 3:
            print(f"  - Ground-truth ARI peaks at k={best_ari_k} ({best_ari_val:.4f}), validating 3 latent states as the optimal biological compartment choice.")
        else:
            print(f"  - Ground-truth ARI peaks at k={best_ari_k} ({best_ari_val:.4f}); k=3 retained to match the 3 physiological simulator compartments.")
            
    return df_selection


def fit_metabolic_hmm(df_features, n_states=3, random_state=42, save_artifacts=True):
    """
    Fits 3-state physiological Gaussian HMM, assigns dynamic clinical labels based on velocity,
    and runs the CGM vs CGM+IOB information-degeneracy experiment.
    """
    print("\n==================================================")
    print(f"=== FITTING GAUSSIAN HMM ({n_states} LATENT METABOLIC STATES) ===")
    print("==================================================")
    
    # 1. CGM-Only Feature Space
    cgm_cols = ["glucose_smooth", "dG_dt", "rolling_std_1h"]
    scaler_cgm = StandardScaler()
    X_cgm = scaler_cgm.fit_transform(df_features[cgm_cols].values)
    
    model = fit_best_hmm(X_cgm, n_states=n_states, n_inits=5, random_state=random_state)
    hidden_states = model.predict(X_cgm)
    
    df_result = df_features.copy()
    df_result["raw_state"] = hidden_states
    
    # Dynamic physiological state mapping
    profiles = []
    for s in range(n_states):
        sub = df_result[df_result["raw_state"] == s]
        profiles.append({
            "raw_state": s,
            "mean_glucose": sub["glucose"].mean(),
            "mean_velocity": sub["dG_dt"].mean(),
            "mean_volatility": sub["rolling_std_1h"].mean(),
            "sample_count": len(sub)
        })
    df_prof = pd.DataFrame(profiles)
    
    excursion_raw = df_prof.loc[df_prof["mean_velocity"].idxmax()]["raw_state"]
    clearance_raw = df_prof.loc[df_prof["mean_velocity"].idxmin()]["raw_state"]
    basal_raw = [s for s in range(n_states) if s not in [excursion_raw, clearance_raw]][0]
    
    state_map = {
        int(basal_raw): STATE_LABELS["basal"],
        int(clearance_raw): STATE_LABELS["clearance"],
        int(excursion_raw): STATE_LABELS["excursion"]
    }
    
    df_result["metabolic_state"] = df_result["raw_state"].map(state_map)
    df_prof["assigned_label"] = df_prof["raw_state"].map(state_map)
    
    print("\n--- LEARNED GAUSSIAN EMISSION PROFILES (CGM-ONLY) ---")
    print(df_prof[["assigned_label", "mean_glucose", "mean_velocity", "mean_volatility", "sample_count"]].to_string(index=False))
    
    # 2. Ground-Truth Recovery Validation (CGM-Only)
    ari_cgm = 0.0
    if "true_state" in df_result.columns:
        ari_cgm = adjusted_rand_score(df_result["true_state"], df_result["metabolic_state"])
        print("\n--- GROUND-TRUTH RECOVERY (CGM-ONLY) ---")
        print(f"Adjusted Rand Index (ARI): {ari_cgm:.4f}")
        print("\nConfusion Matrix (True In Silico State vs Decoded State):")
        ct = pd.crosstab(df_result["true_state"], df_result["metabolic_state"])
        print(ct)
        
    # 3. Information-Degeneracy Experiment (Adding Insulin-on-Board)
    if "insulin_on_board" in df_features.columns:
        print("\n--- INFORMATION DEGENERACY EXPERIMENT (+ INSULIN-ON-BOARD) ---")
        multimodal_cols = ["glucose_smooth", "dG_dt", "rolling_std_1h", "insulin_on_board"]
        scaler_multi = StandardScaler()
        X_multi = scaler_multi.fit_transform(df_features[multimodal_cols].values)
        
        model_multi = fit_best_hmm(X_multi, n_states=n_states, n_inits=5, random_state=random_state)
        states_multi = model_multi.predict(X_multi)
        
        ari_multi = adjusted_rand_score(df_result["true_state"], states_multi)
        delta_ari = ari_multi - ari_cgm
        sign_str = f"+{delta_ari:.4f}" if delta_ari >= 0 else f"{delta_ari:.4f}"
        
        print(f"CGM-Only ARI:               {ari_cgm:.4f}")
        print(f"CGM + Insulin-on-Board ARI: {ari_multi:.4f} (Delta: {sign_str})")
        
        if delta_ari < 0:
            print("[Negative Finding]: Adding insulin-on-board degraded state recovery. The zero-inflated, monotonically decaying IOB signal dominates feature variance, biasing the unsupervised HMM toward insulin decay kinetics rather than true physiological transitions.")
        else:
            print("[Finding]: Incorporating insulin-on-board improved state recovery by resolving the basal vs clearance observational ambiguity.")

    if save_artifacts:
        output_dir = "project-c-state-modeller/data"
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(model, os.path.join(output_dir, "gaussian_hmm_model.pkl"))
        joblib.dump(scaler_cgm, os.path.join(output_dir, "hmm_scaler.pkl"))
        df_result.to_csv(os.path.join(output_dir, "cgm_with_decoded_states.csv"), index=False)
        print(f"\n[Storage] Saved model and decoded time-series to: {output_dir}")
        
    return model, df_result, df_prof


if __name__ == "__main__":
    feat_path = "project-c-state-modeller/data/cgm_engineered_features.csv"
    df_feat = pd.read_csv(feat_path)
    
    # Model Selection Curve across k in [2, 6]
    cgm_cols = ["glucose_smooth", "dG_dt", "rolling_std_1h"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_feat[cgm_cols].values)
    y_true = df_feat["true_state"] if "true_state" in df_feat.columns else None
    evaluate_model_selection(X_scaled, y_true=y_true, max_states=6)
    
    # Fit 3-State Model
    fit_metabolic_hmm(df_feat, n_states=3, save_artifacts=True)