import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP
from align_datasets import align_three_datasets


def generate_batch_effect_plots(aligned_matrices, labels_dict):
    """Concatenate aligned matrices for joint PCA/UMAP to demonstrate tissue/platform dominance."""
    
    print("=== BUILDING JOINT CONCATENATED MATRIX  ===")
    
    dfs_to_concat = []
    metadata_list = []
    
    dataset_tissues = {
        "GSE18732": "Skeletal Muscle",
        "GSE10946": "Ovarian Cumulus Cells",
        "GSE89632": "Liver Tissue"
    }
    
    for name, df in aligned_matrices.items():
        dfs_to_concat.append(df)
        y = labels_dict[name]
        
        for sample_id in df.columns:
            metadata_list.append({
                "Sample_ID": sample_id,
                "Dataset": name,
                "Tissue": dataset_tissues[name],
                "Phenotype": str(y.loc[sample_id])
            })
            
    # Joint matrix: Genes x Total Patients (10369 x 204)
    X_joint = pd.concat(dfs_to_concat, axis=1)
    df_meta = pd.DataFrame(metadata_list).set_index("Sample_ID")
    
    print(f"Joint Concatenated Matrix Shape: {X_joint.shape} (Genes x Total Samples)")
    
    # Transpose to Samples x Genes (204 x 10369)
    X_samples = X_joint.T.values
    
    # 1. Run PCA
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_samples)
    df_meta["PCA1"] = pca_coords[:, 0]
    df_meta["PCA2"] = pca_coords[:, 1]
    var_exp = pca.explained_variance_ratio_ * 100
    
    # 2. Run UMAP
    print("Running UMAP...")
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_coords = umap_model.fit_transform(X_samples)
    df_meta["UMAP1"] = umap_coords[:, 0]
    df_meta["UMAP2"] = umap_coords[:, 1]
    
    # 3. Plot Figures (Hue = Tissue, Style = Phenotype)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot A: Joint PCA
    sns.scatterplot(
        data=df_meta, x="PCA1", y="PCA2", hue="Tissue", style="Phenotype",
        s=90, alpha=0.85, ax=axes[0], palette="Set2"
    )
    axes[0].set_title(
        f"Joint PCA Space (Tissue / Platform Dominance)\nPC1: {var_exp[0]:.1f}% | PC2: {var_exp[1]:.1f}%", 
        fontsize=12, fontweight="bold"
    )
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot B: Joint UMAP
    sns.scatterplot(
        data=df_meta, x="UMAP1", y="UMAP2", hue="Tissue", style="Phenotype",
        s=90, alpha=0.85, ax=axes[1], palette="Set2"
    )
    axes[1].set_title("Joint UMAP Embedding (Platform / Confounded Clusters)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("UMAP Dimension 1")
    axes[1].set_ylabel("UMAP Dimension 2")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    # Explicit figure caption addressing the confounder
    fig.text(
        0.5, -0.02, 
        "Note: Tissue, study, and array platform are perfectly confounded in this design; "
        "the separation reflects their combined effect and cannot be attributed to tissue biology alone.",
        ha="center", fontsize=10, fontstyle="italic"
    )
    
    plt.tight_layout()
    
    # Ensure figures/ directory exists
    figures_dir = "project-a-convergence-classifier/figures"
    os.makedirs(figures_dir, exist_ok=True)
    output_fig_path = os.path.join(figures_dir, "tissue_dominance_batch_effects.png")
    
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    print(f"\n[SUCCESS] Figure saved to: {output_fig_path}")
    plt.close()

if __name__ == "__main__":
    paths = {
        "GSE18732": "project-a-convergence-classifier/data/GSE18732_series_matrix.txt",
        "GSE10946": "project-a-convergence-classifier/data/GSE10946_series_matrix.txt",
        "GSE89632": "project-a-convergence-classifier/data/GSE89632_series_matrix.txt"
    }
    biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
    gpl570 = "project-a-convergence-classifier/data/GPL570.annot"
    gpl14951 = "project-a-convergence-classifier/data/GPL14951.annot"

    matrices, labels, gene_intersection = align_three_datasets(paths, biomart_path, gpl570, gpl14951)
    generate_batch_effect_plots(matrices, labels)