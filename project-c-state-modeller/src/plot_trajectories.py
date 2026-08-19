import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from evaluate_states import compute_state_transition_matrix

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})


def plot_patient_state_timeline(df_decoded, output_path="project-c-state-modeller/figures/simulated_patient_states.png", days_to_plot=3):
    """
    Plots in silico CGM trajectory color-coded by the active decoded state.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = df_decoded.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    start_ts = df["timestamp"].min()
    end_ts = start_ts + pd.Timedelta(days=days_to_plot)
    df_sub = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    
    # Target Range (70-180 mg/dL)
    ax.axhspan(70, 180, color="#22c55e", alpha=0.10, label="Target Range (70-180 mg/dL)")
    
    color_map = {
        "State 0: Fasting Basal": "#2563eb",         # Royal Blue
        "State 1: Insulin Clearance": "#059669",     # Emerald Green
        "State 2: Absorption Excursion": "#d97706"  # Amber
    }
    
    for i in range(len(df_sub) - 1):
        t_start = df_sub["timestamp"].iloc[i]
        t_end = df_sub["timestamp"].iloc[i+1]
        g_start = df_sub["glucose"].iloc[i]
        g_end = df_sub["glucose"].iloc[i+1]
        st_name = df_sub["metabolic_state"].iloc[i]
        
        ax.plot([t_start, t_end], [g_start, g_end], color=color_map.get(st_name, "#64748b"), linewidth=2.2)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2563eb", lw=2.5, label="State 0: Fasting Basal"),
        Line2D([0], [0], color="#059669", lw=2.5, label="State 1: Insulin Clearance"),
        Line2D([0], [0], color="#d97706", lw=2.5, label="State 2: Absorption Excursion"),
        Line2D([0], [0], color="#22c55e", alpha=0.3, lw=8, label="Target Range (70-180 mg/dL)")
    ]
    
    ax.set_title("In Silico Validation: Decoded Latent Metabolic States (72-Hour Window)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Date & Time (MM-DD HH:MM)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sensor Glucose (mg/dL)", fontsize=11, fontweight="bold")
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    ax.set_ylim(70, 225)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualisation Ready] Saved trajectory timeline to: {output_path}")


def plot_transition_matrix_heatmap(df_decoded, output_path="project-c-state-modeller/figures/state_transition_heatmap.png"):
    """
    Renders transition heatmap strictly reordered to semantic order (State 0 -> State 2 -> State 1).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trans_mat, _ = compute_state_transition_matrix(df_decoded)
    
    state_names = dict(zip(df_decoded["raw_state"], df_decoded["metabolic_state"]))
    
    # Semantic ordering: Basal (0) -> Excursion (2) -> Clearance (1)
    basal_raw = [k for k, v in state_names.items() if "Fasting Basal" in v][0]
    excursion_raw = [k for k, v in state_names.items() if "Absorption Excursion" in v][0]
    clearance_raw = [k for k, v in state_names.items() if "Insulin Clearance" in v][0]
    
    ordered_indices = [basal_raw, excursion_raw, clearance_raw]
    ordered_labels = ["State 0\n(Basal)", "State 2\n(Excursion)", "State 1\n(Clearance)"]
    
    # Reindex transition matrix to match semantic order
    reordered_mat = trans_mat[np.ix_(ordered_indices, ordered_indices)]
    
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(
        reordered_mat,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=ordered_labels,
        yticklabels=ordered_labels,
        cbar_kws={"label": "5-Minute Transition Probability P(S_t+1 | S_t)"},
        linewidths=1.2,
        linecolor="white",
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title("Empirical Metabolic State Transition Probability Matrix", fontsize=12, fontweight="bold", pad=14)
    plt.xlabel("Next State (t + 5 min)", fontsize=10, fontweight="bold")
    plt.ylabel("Current State (t)", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualisation Ready] Saved transition heatmap to: {output_path}")


if __name__ == "__main__":
    data_path = "project-c-state-modeller/data/cgm_with_decoded_states.csv"
    df_decoded = pd.read_csv(data_path)
    plot_patient_state_timeline(df_decoded)
    plot_transition_matrix_heatmap(df_decoded)