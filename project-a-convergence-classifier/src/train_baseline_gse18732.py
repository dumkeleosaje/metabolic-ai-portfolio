import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from align_datasets import align_three_datasets

warnings.filterwarnings("ignore")


def run_gse18732_elasticnet_baseline():
    print("\n==================================================")
    print("=== GSE18732 BASELINE: ELASTICNET + STRICT SPARSITY ===")
    print("==================================================")
    
    # 1. Load Aligned Feature Space (10,369 Genes)
    paths = {
        "GSE18732": "project-a-convergence-classifier/data/GSE18732_series_matrix.txt",
        "GSE10946": "project-a-convergence-classifier/data/GSE10946_series_matrix.txt",
        "GSE89632": "project-a-convergence-classifier/data/GSE89632_series_matrix.txt"
    }
    biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
    gpl570 = "project-a-convergence-classifier/data/GPL570.annot"
    gpl14951 = "project-a-convergence-classifier/data/GPL14951.annot"

    matrices, labels, _ = align_three_datasets(paths, biomart_path, gpl570, gpl14951)
    
    # Extract GSE18732
    X_df = matrices["GSE18732"]  # Genes x Patients (10369 x 118)
    y_series = labels["GSE18732"]
    
    # Binary Filter: NGT vs DM
    binary_mask = y_series.isin(["NGT", "DM"])
    X_binary = X_df.loc[:, binary_mask].T  # Samples x Genes (106 x 10369)
    y_binary = (y_series[binary_mask] == "DM").astype(int)
    
    print(f"Dataset Shape (NGT vs DM): {X_binary.shape[0]} samples across {X_binary.shape[1]} shared genes.")
    print("Class Distribution:\n", y_binary.value_counts())

    # 2. Outer Stratified 5-Fold Cross-Validation
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracies, aucs, selected_gene_counts = [], [], []
    selected_genes_per_fold = []
    
    # STRICT SPARSITY PARAMETER GRID
    # Lower C values enforce stronger regularization.
    # Higher l1_ratio values force L1 penalty (LASSO) to drop uninformative features to zero.
    param_grid = {
        "C": [0.005, 0.01, 0.05, 0.1, 0.2],
        "l1_ratio": [0.5, 0.7, 0.9, 1.0]
    }

    print("\n--- Running 5-Fold Outer CV with Inner GridSearch ---")

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_binary, y_binary), 1):
        X_train, X_test = X_binary.iloc[train_idx], X_binary.iloc[test_idx]
        y_train, y_test = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
        
        # Base Model: ElasticNet with SAGA solver
        base_model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=3000,
            random_state=42,
            class_weight="balanced"
        )
        
        # Inner 3-Fold Grid Search
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Evaluate on Held-out Test Fold
        preds = best_model.predict(X_test)
        probs = best_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        # Track selected genes
        coefs = best_model.coef_[0]
        active_indices = np.where(coefs != 0)[0]
        active_genes = X_binary.columns[active_indices].tolist()
        
        accuracies.append(acc)
        aucs.append(auc)
        selected_gene_counts.append(len(active_genes))
        selected_genes_per_fold.append(active_genes)
        
        print(
            f"Fold {fold}: Best C={grid_search.best_params_['C']}, "
            f"l1_ratio={grid_search.best_params_['l1_ratio']} | "
            f"Acc = {acc:.3f} | ROC-AUC = {auc:.3f} | Active Genes = {len(active_genes)}"
        )

    print("\n==================================================")
    print("=== SUMMARY PERFORMANCE (STRICT ELASTICNET) ===")
    print("==================================================")
    print(f"Mean Accuracy:           {np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}")
    print(f"Mean ROC-AUC:            {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    print(f"Mean Selected Features:  {np.mean(selected_gene_counts):.1f}")
    
    # 3. Identify Consistently Selected Genes Across Folds
    all_genes = [gene for fold_list in selected_genes_per_fold for gene in fold_list]
    gene_counts = pd.Series(all_genes).value_counts()
    print("\n--- Top Genes Selected Across Multiple CV Folds ---")
    print(gene_counts.head(15))


if __name__ == "__main__":
    run_gse18732_elasticnet_baseline()