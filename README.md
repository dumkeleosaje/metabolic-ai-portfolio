# Metabolic AI Portfolio

Three projects I built over summer 2026, looking at one question from different angles: is chronic hyperinsulinaemia (persistently high insulin) a shared upstream driver of type 2 diabetes, PCOS and fatty liver disease?

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/dumkeleosaje/metabolic-ai-portfolio)](https://github.com/dumkeleosaje/metabolic-ai-portfolio)

---

## Why I looked at this

Type 2 diabetes, PCOS and non-alcoholic fatty liver disease (NAFLD) are usually treated by different specialists as separate conditions. But they turn up together far more often than chance would predict, they respond to the same metabolic interventions, and all three involve insulin resistance. Reaven made a version of this argument back in 1988.

I wanted to see what that idea looks like in real data. If these diseases share a root cause, does that show up as a shared gene expression signature? Does the literature agree on the mechanisms involved? And can you pick out metabolic "states" from glucose data alone?

The short answer from Project A is that the shared signal is much weaker than I expected. Each tissue seems to respond to insulin resistance in its own way. I think that negative result is the most interesting thing in the portfolio, so I've tried to be upfront about it below.

---

## Projects

### Project A: Cross-tissue transcriptomic classifier

**Question:** Do T2D, PCOS and NAFLD share a gene expression signature across muscle, ovary and liver?

**Data:** Three public microarray datasets from NCBI GEO: GSE18732 (skeletal muscle, T2D), GSE10946 (ovary, PCOS) and GSE89632 (liver, NAFLD). Each comes from a different array platform.

**What I did:**
- Built a pipeline to parse each dataset, map probes to genes (Ensembl BioMart and GEO platform files), log-transform, and align all three to 10,369 shared genes.
- Ran PCA and UMAP on the combined data. Tissue completely dominated: PC1 explained 60.2% of the variance, against 7.4% within any single tissue. A model trained on pooled data would just learn tissue type.
- So instead I trained ElasticNet classifiers on one tissue and tested them on another, with nested cross-validation and feature selection kept inside each fold to avoid leakage. Switching from per-sample to per-gene standardisation lifted muscle-to-ovary AUC from 0.417 to 0.705.
- Tested all six transfer directions against permutation nulls with Benjamini-Hochberg correction.

**What I found:**
- No cross-tissue transfer survived FDR correction (all q ≥ 0.196).
- One direction looked significant at first (p = 0.046). A matched sample-size control showed it was a power artefact rather than real biology, because it tracked the size of the target dataset.
- Genome-wide agreement between tissues was essentially zero. Restricting to the 2,000 most variable genes showed a weak but significant muscle-liver agreement (r = 0.070).
- Within muscle, gene expression seemed to predict insulin resistance (HOMA-IR). When I checked for confounding, BMI explained it: after adjusting for BMI, age and sex the signal disappeared.

**Takeaway:** The data argue against the strong version of the hypothesis (one shared transcriptional programme). They're consistent with the weaker version, where high insulin is a shared upstream trigger but each tissue responds differently. The main limitations are small cohorts (n = 23 for PCOS, n = 43 for liver), and that tissue, study and platform are perfectly confounded in this design.

→ [View Project A](project-a-convergence-classifier/)

---

### Project B: Autophagy-senescence knowledge graph

**Question:** Where does the research literature on autophagy and cellular senescence contradict itself?

**Data:** 295 PubMed abstracts retrieved through the Entrez API.

**What I did:**
- Extracted biomedical entities with SciSpacy and built 57,079 sentence-level co-occurrence pairs.
- Built a NetworkX graph (456 entities, 745 relationships, each supported by at least 3 papers) and classified each relationship as activating, inhibiting or neutral. I only looked at the verbs between the two entities, and treated negations as neutral.
- Counted a relationship as "contested" only when different papers clearly disagreed on its direction.

**What I found:** 119 contested relationships. The number started at 402 and came down through four rounds of fixing my own counting errors, which taught me more than the final figure did. Half-sample robustness checks gave a degree rank correlation of 0.80, and the main contested edge (autophagy and senescence) held up.

**Limitations:** Co-occurrence isn't the same as interaction, keyword classification is a heuristic (dependency parsing would be the proper fix), and 295 abstracts is a small corpus.

→ [View Project B](project-b-knowledge-graph/)

---

### Project C: Metabolic state modeller

**Question:** Can a Hidden Markov Model recover hidden metabolic states from continuous glucose monitor (CGM) readings alone?

**Data:** A physiological CGM simulator I wrote, with known ground-truth states so the model can be checked properly.

**What I did:**
- Engineered time-series features (smoothed glucose, rate of change, rolling variability) and fitted Gaussian HMMs with hmmlearn.
- Compared BIC and Adjusted Rand Index (ARI) for choosing the number of states.
- Analysed transition probabilities, how long the model stayed in each state compared with a Markov null, recovery after meals, and standard clinical glucose metrics for each state.

**What I found:**
- With no labels, the model recovered three states (fasting, meal absorption and insulin clearance) with an ARI of 0.775.
- BIC kept improving all the way up to six states, while ARI peaked at three. BIC tends to over-split continuous signals, so I chose k = 3 on interpretability.
- Adding insulin-on-board as a feature made state recovery worse, which I hadn't expected.

**Limitations:** It's simulated data from a single synthetic patient, so recovery times are more regular than real CGM data would be. Testing it on real data (OhioT1DM) is the next step.

→ [View Project C](project-c-state-modeller/)

---

## Tools

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| Data and statistics | pandas, NumPy, SciPy |
| Machine learning | scikit-learn, hmmlearn, UMAP |
| Bioinformatics and NLP | Biopython, SciSpacy, NetworkX, Pyvis |
| Testing | pytest |

---

## What I'd do next

- Move Project A from single genes to pathway-level scores (ssGSEA or GSVA). Pathways may agree across tissues even where individual genes don't.
- Re-run Project A on larger RNA-seq cohorts with better clinical metadata, especially for PCOS.
- Replace keyword matching in Project B with dependency parsing.
- Validate Project C on real CGM data.

---

## Background reading

- Reaven (1988): Role of insulin resistance in human disease
- Laplante & Sabatini (2012): mTOR signalling and the control of cell metabolism
- López-Otín et al. (2013, 2023): Hallmarks of Aging
- Lean et al. (2018): Primary care-led weight management for remission of type 2 diabetes (DiRECT)

---

## About me

I'm a second-year Computer Science and AI student at the University of Bath, on the Integrated Master's with a year in industry. I'm interested in using machine learning on biomedical data, especially metabolic disease. I built these projects in my own time over the summer, mostly to learn how to do computational research properly: validating results, testing against nulls, and being honest when something doesn't work.

[GitHub](https://github.com/dumkeleosaje) · [LinkedIn](https://linkedin.com/in/dumkeleosaje) · [Website](https://dumkeleosaje.github.io)
