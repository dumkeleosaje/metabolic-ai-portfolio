import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from align_datasets import align_three_datasets
from clean_geo import standardise_per_gene

warnings.filterwarnings("ignore")


def run_honest_gse18732_benchmark():
    print("\n==================================================")
    print("=== GSE18732 HONEST BENCHMARK (PER-GENE STD) ===")
    print("==================================================")
    
    paths = {
        "GSE18732": "project-a-convergence-classifier/data/GSE18732_series_matrix.txt",
        "GSE10946": "project-a-convergence-classifier/data/GSE10946_series_matrix.txt",
        "GSE89632": "project-a-convergence-classifier/data/GSE89632_series_matrix.txt"
    }
    biomart_path = "project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"
    gpl570 = "project-a-convergence-classifier/data/GPL570.annot"
    gpl14951 = "project-a-convergence-classifier/data/GPL14951.annot"

    # Load 10,369 aligned raw/log2 matrices
    matrices, labels, _ = align_three_datasets(paths, biomart_path, gpl570, gpl14951)
    
    # Perform PER-GENE standardisation within GSE18732
    X_raw_genes = matrices["GSE18732"] # Genes x Patients
    X_std_genes = standardise_per_gene(X_raw_genes, step_label="GSE18732 Per-Gene Standardisation")
    
    y_series = labels["GSE18732"]
    binary_mask = y_series.isin(["NGT", "DM"])
    
    X_binary = X_std_genes.loc[:, binary_mask].T  # Samples x Genes (106 x 10369)
    y_binary = (y_series[binary_mask] == "DM").astype(int)
    
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        "ElasticNet (Sparsity)": {"acc": [], "auc": []},
        "Filtered ElasticNet (Top 500)": {"acc": [], "auc": []},
        "Random Forest (Top 500)": {"acc": [], "auc": []},
        "SVM RBF (Top 500)": {"acc": [], "auc": []}
    }

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_binary, y_binary), 1):
        X_train, X_test = X_binary.iloc[train_idx], X_binary.iloc[test_idx]
        y_train, y_test = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
        
        # M1: ElasticNet Unfiltered
        m1 = LogisticRegression(penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.7, max_iter=2000, random_state=42, class_weight="balanced")
        m1.fit(X_train, y_train)
        results["ElasticNet (Sparsity)"]["acc"].append(accuracy_score(y_test, m1.predict(X_test)))
        results["ElasticNet (Sparsity)"]["auc"].append(roc_auc_score(y_test, m1.predict_proba(X_test)[:, 1]))
        
        # ANOVA Feature Selection on Train Fold ONLY
        selector = SelectKBest(score_func=f_classif, k=500)
        X_train_sub = selector.fit_transform(X_train, y_train)
        X_test_sub = selector.transform(X_test)
        
        # M2: Filtered ElasticNet
        m2 = LogisticRegression(penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5, max_iter=2000, random_state=42, class_weight="balanced")
        m2.fit(X_train_sub, y_train)
        results["Filtered ElasticNet (Top 500)"]["acc"].append(accuracy_score(y_test, m2.predict(X_test_sub)))
        results["Filtered ElasticNet (Top 500)"]["auc"].append(roc_auc_score(y_test, m2.predict_proba(X_test_sub)[:, 1]))
        
        # M3: Random Forest
        m3 = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight="balanced")
        m3.fit(X_train_sub, y_train)
        results["Random Forest (Top 500)"]["acc"].append(accuracy_score(y_test, m3.predict(X_test_sub)))
        results["Random Forest (Top 500)"]["auc"].append(roc_auc_score(y_test, m3.predict_proba(X_test_sub)[:, 1]))

        # M4: SVM RBF
        m4 = SVC(kernel="rbf", C=1.0, probability=True, random_state=42, class_weight="balanced")
        m4.fit(X_train_sub, y_train)
        results["SVM RBF (Top 500)"]["acc"].append(accuracy_score(y_test, m4.predict(X_test_sub)))
        results["SVM RBF (Top 500)"]["auc"].append(roc_auc_score(y_test, m4.predict_proba(X_test_sub)[:, 1]))

    print("\n==================================================")
    print("=== HONEST PER-GENE BENCHMARK TABLE ===")
    print("==================================================")
    for model_name, metrics in results.items():
        mean_acc = np.mean(metrics["acc"])
        std_acc = np.std(metrics["acc"])
        mean_auc = np.mean(metrics["auc"])
        std_auc = np.std(metrics["auc"])
        print(f"{model_name:<35} | Acc: {mean_acc:.3f} +/- {std_acc:.3f} | ROC-AUC: {mean_auc:.3f} +/- {std_auc:.3f}")

if __name__ == "__main__":
    run_honest_gse18732_benchmark()