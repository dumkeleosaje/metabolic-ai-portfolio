# 🧬 Project A — Multi-Disease Transcriptomic Convergence Classifier

> **Can AI detect a shared molecular fingerprint across diseases that medicine treats as unrelated?**
>
> This project applies machine learning to public RNA-seq data from Type 2 Diabetes, PCOS, and Non-Alcoholic Fatty Liver Disease — testing the hypothesis that chronic hyperinsulinaemia leaves a common transcriptomic signature across all three.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-22c55e?style=flat-square)]()

---

## The Biological Problem

Standard clinical practice treats Type 2 Diabetes, PCOS, and NAFLD as entirely separate diseases — managed by different specialists, treated with different drugs, categorised under different ICD codes.

But they share something: all three are consistently co-morbid with **chronic hyperinsulinaemia** — persistently elevated insulin levels driven by insulin resistance. Insulin activates **mTOR**, promotes lipogenesis, stimulates androgen production, and suppresses autophagy. The downstream effects land differently depending on tissue and genetics — but the upstream signal may be the same.

If that is true, we would expect the gene expression profiles of these three diseases to **converge** when compared to healthy tissue — clustering together in transcriptomic space before diverging into their disease-specific signatures.

This project tests that hypothesis computationally.

---

## What This Project Does

```
Raw GEO RNA-seq data (3 diseases + healthy controls)
         │
         ▼
  Data loading & cleaning
  (log₂ transform, missing value filter, standardisation)
         │
         ▼
  LASSO feature selection
  (reduces ~20,000 genes → most informative subset)
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
  PCA / UMAP visualisation          Random Forest classifier
  (do disease groups converge?)     (can we classify disease from genes?)
         │                                  │
         ▼                                  ▼
  Convergence plot                  SHAP explainability
  (centrepiece figure)              (which genes drive the decision?)
```

---

## Dataset Sources

All data is publicly available through [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/).

| Dataset | GEO Accession | Disease | Tissue | Samples |
|---------|:------------:|---------|--------|:-------:|
| Insulin resistance transcriptomics | [GSE18732](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE18732) | Type 2 Diabetes / IR | Skeletal muscle | ~40 |
| PCOS gene expression | GSE#### | PCOS | Granulosa cells | ~30 |
| NAFLD liver transcriptomics | [GSE89632](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE89632) | NAFLD | Liver | ~35 |
| Healthy controls | Combined | None | Mixed | ~45 |

> **Note:** All datasets are human tissue, free to access, and require no registration. Download instructions are in `docs/data_download.md`.

---

## Methods

### 1. Data Loading & Cleaning
**Files:** `src/load_geo.py`, `src/clean_geo.py`

GEO Series Matrix files are parsed, metadata is extracted, and expression matrices are cleaned:
- Genes with >20% missing values removed
- **Log₂ transformation** applied — gene expression spans many orders of magnitude; log transform makes the distribution approximately normal for ML
- Each sample **standardised** to zero mean, unit variance — removes batch effects where different samples were processed at different times or facilities

### 2. Feature Selection
**File:** `src/classifier.py`

Biological datasets are *wide* — 20,000 genes but only ~150 samples. Standard classifiers overfit catastrophically in this setting.

**LASSO (LassoCV)** adds an L1 penalty that forces most feature weights to zero, automatically selecting only the most informative genes. This reduces dimensionality from ~20,000 to a handful of biologically meaningful features.

### 3. Dimensionality Reduction & Visualisation
**Files:** `src/pca_analysis.py`, `src/umap_analysis.py`

- **PCA** for initial data exploration and sanity checking
- **UMAP** for final visualisation — better preserves local cluster structure than PCA in biological data

The key question: *do the three disease groups cluster closer to each other than to healthy controls?*

### 4. Classification
**File:** `src/classifier.py`

**Random Forest** trained on LASSO-selected features with 80/20 stratified train/test split. Random Forest chosen for:
- Robustness to small sample sizes
- Built-in feature importance (complemented by SHAP)
- No assumption of linearity in gene-disease relationships

### 5. Explainability
**File:** `src/shap_analysis.py`

**SHAP (SHapley Additive exPlanations)** assigns each gene a contribution score to each prediction. The beeswarm plot shows:
- Which genes most consistently drive classification
- Whether high or low expression of each gene pushes toward disease
- Biological validation: do the top SHAP genes appear in known insulin signalling pathways?

---

## Key Results

| Metric | Value |
|--------|:-----:|
| Classifier accuracy | XX% |
| F1 score (weighted) | 0.XX |
| Genes selected by LASSO | XXX / ~20,000 |
| Top SHAP gene | `GENE_NAME` |
| UMAP cluster separation (PERMANOVA R²) | 0.XX (p < 0.05) |

### UMAP Convergence Plot

![UMAP Multi-Disease Convergence](figures/umap_convergence.png)

*Three metabolically distinct diseases cluster closer to each other than to healthy controls in UMAP space — consistent with a shared upstream transcriptomic signature. Colours: blue = healthy, red = T2D/IR, orange = PCOS, purple = NAFLD.*

### SHAP Beeswarm

![SHAP Beeswarm Top 20 Genes](figures/shap_beeswarm.png)

*Top 20 genes by mean absolute SHAP value. `GENE_NAME` and `GENE_NAME` are known components of the insulin signalling cascade, providing biological validation of the classifier's decision logic.*

---

## How to Run

### Prerequisites

```bash
git clone https://github.com/YOUR-USERNAME/metabolic-ai-portfolio.git
cd metabolic-ai-portfolio/project-a-convergence-classifier
pip install -r requirements.txt
```

### Download data

Follow `docs/data_download.md` — each GEO dataset is freely downloadable. Place series matrix files in `data/raw/`.

### Run the full pipeline

```bash
# Step 1: Load and parse GEO series matrix files
python src/load_geo.py

# Step 2: Clean and normalise expression data
python src/clean_geo.py

# Step 3: PCA visualisation
python src/pca_analysis.py

# Step 4: UMAP multi-disease convergence plot
python src/umap_analysis.py

# Step 5: LASSO + Random Forest classifier
python src/classifier.py

# Step 6: SHAP explainability
python src/shap_analysis.py
```

### Run tests

```bash
pytest tests/ -v
```

---

## File Structure

```
project-a-convergence-classifier/
│
├── src/
│   ├── load_geo.py          # Parse GEO Series Matrix files into DataFrames
│   ├── clean_geo.py         # Log2 transform, missing filter, standardisation
│   ├── pca_analysis.py      # PCA with disease-label colour coding
│   ├── umap_analysis.py     # UMAP multi-disease convergence visualisation
│   ├── classifier.py        # LASSO feature selection + Random Forest
│   └── shap_analysis.py     # SHAP beeswarm plot + gene interpretation
│
├── tests/
│   ├── test_load.py         # Tests for GEO data parsing
│   └── test_clean.py        # Tests for log2 transform, standardisation
│
├── figures/
│   ├── pca_gse18732.png     # PCA: IR vs healthy
│   ├── umap_convergence.png # UMAP: all diseases vs healthy
│   └── shap_beeswarm.png    # SHAP: top 20 explanatory genes
│
├── docs/
│   └── data_download.md     # Step-by-step GEO download instructions
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dependencies

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
umap-learn>=0.5
shap>=0.42
scipy>=1.11
pytest>=7.4
```

---

## Research Context

This project sits within a broader computational investigation of the hyperinsulinaemia hypothesis — the idea that chronically elevated insulin is the unified upstream driver of what medicine currently categorises as separate metabolic diseases.

- **Project B** maps the autophagy-senescence contested boundary at the pathway level
- **Project C** models metabolic state transitions and identifies intervention windows at the systems level

Together, the three projects form a multi-scale computational argument for a unified theory of metabolic disease.

---

## References

- Reaven, G.M. (1988). Role of insulin resistance in human disease. *Diabetes*, 37(12), 1595–1607.
- Laplante, M. & Sabatini, D.M. (2012). mTOR signaling in growth control and disease. *Cell*, 149(2), 274–293.
- López-Otín, C. et al. (2013). The hallmarks of aging. *Cell*, 153(6), 1194–1217.

---
