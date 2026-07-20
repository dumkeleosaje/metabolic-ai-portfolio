# Metabolic AI Portfolio

> Computational investigation of chronic hyperinsulinaemia as a unified 
> upstream driver of metabolic disease.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/your-username/metabolic-ai-portfolio)](https://github.com/your-username/metabolic-ai-portfolio)

---

## The Research Question

By the time a doctor diagnoses Type 2 diabetes, PCOS, or non-alcoholic 
fatty liver disease, the underlying cause — chronic hyperinsulinaemia — 
has been present for 10–15 years. These conditions are treated as separate 
diseases by separate specialists, but growing evidence suggests they share 
a single upstream molecular driver.

This portfolio uses AI and computational biology to investigate that 
hypothesis across three biological scales.

---

## Projects

### Project A — Multi-Disease Transcriptomic Convergence Classifier
**Question:** Do Type 2 diabetes, PCOS, and NAFLD share a common 
transcriptomic signature that diverges from healthy tissue?

**Methods:** LASSO feature selection, PCA/UMAP dimensionality reduction, 
Random Forest classification, SHAP explainability

**Data:** NCBI GEO public RNA-seq datasets (GSE18732, GSE89632)

**Key finding:** UMAP visualisation reveals that all three disease groups 
cluster closer to each other than to healthy controls, suggesting shared 
upstream molecular pathology.

→ [View Project A](project-a-convergence-classifier/)

---

### Project B — Autophagy-Senescence Knowledge Graph
**Question:** Where is the contested molecular boundary between 
healthy fasting-induced autophagy and pathological senescent-survival 
autophagy?

**Methods:** PubMed API, SciSpacy NER, NetworkX, BioBERT semantic clustering

**Data:** 400 PubMed abstracts on autophagy and senescence

**Key finding:** 5 protein relationships identified where the literature 
contains direct contradictions about direction of effect — representing 
the biological uncertainty around the senescence threshold.

→ [View Project B](project-b-knowledge-graph/)

---

### Project C — Metabolic State Trajectory Modeller
**Question:** Can Hidden Markov Models identify the precise transition 
moments between metabolic flexibility and rigidity from continuous 
glucose monitor data?

**Methods:** Gaussian HMM, feature engineering on CGM data, causal 
inference with DoWhy

**Data:** OhioT1DM open dataset (8 patients, 8 weeks of CGM data)

**Key finding:** State transitions to metabolic rigidity correlate 
significantly with meal carbohydrate content (r = X.XX, p < 0.05), 
identifying windows of maximum intervention leverage.

→ [View Project C](project-c-state-modeller/)

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Languages | Python 3.12, C, Rust |
| ML/DL | PyTorch, scikit-learn, hmmlearn |
| Bioinformatics | SciSpacy, Biopython, NetworkX |
| Data | pandas, NumPy, UMAP |
| Deployment | Flask, Docker, Hugging Face Spaces |
| Testing | pytest, GitHub Actions |

---

## Background Reading

Papers that ground this work:

- Reaven (1988) — Role of insulin resistance in human disease
- Laplante & Sabatini (2012) — mTOR signalling and the control of cell metabolism
- López-Otín et al. (2013, 2023) — Hallmarks of Aging
- Lean et al. (2018) — Primary care-led weight management for T2D reversal (DiRECT)

---

## About

First-year CS & AI student at the University of Bath (BSc Integrated Masters), 
researching the computational biology of metabolic disease.

Research interests: insulin resistance as a unified disease mechanism, 
autophagy-senescence threshold modelling, metabolic state trajectory control.

[GitHub](https://github.com/dumkeleosaje) · 
[LinkedIn](https://linkedin.com/in/dumkeleosaje) · 
[Website](https://dumkeleosaje.github.io)
