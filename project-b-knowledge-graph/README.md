# 🔬 Project B — Autophagy-Senescence Knowledge Graph

> **Where exactly does healthy cellular recycling become pathological zombie-cell survival?**
>
> This project builds an NLP-driven biological knowledge graph from 400+ PubMed abstracts to map the protein interaction network at the contested boundary between autophagy and cellular senescence — identifying the molecular relationships where the scientific literature itself disagrees.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![spaCy](https://img.shields.io/badge/spaCy-3.6-09A3D5?style=flat-square&logo=spacy&logoColor=white)](https://spacy.io)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.1-4B8BBE?style=flat-square)](https://networkx.org)
[![HuggingFace](https://img.shields.io/badge/🤗-BioBERT-FFD21E?style=flat-square)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

---

## The Biological Problem

**Autophagy** is the cell's self-cleaning process — triggered by fasting and nutrient deprivation, it breaks down damaged proteins and organelles, recycling their components. It is one of the most important mechanisms in longevity research.

**Cellular senescence** is what happens when a cell is too damaged to safely divide. Instead of dying (apoptosis), it enters a zombie state — permanently arrested, but secreting a toxic cocktail of inflammatory signals called the **SASP** (Senescence-Associated Secretory Phenotype). Senescent cells accumulate with age and are implicated in cancer, Alzheimer's disease, and metabolic dysfunction.

The relationship between these two processes is **not fully understood**. Recent research shows that:
- In healthy cells, autophagy clears the damage that would otherwise trigger senescence
- In already-senescent cells, autophagy is *hijacked* to help the zombie cell survive
- The molecular threshold at which protective autophagy becomes senescence-supporting is unknown

**This threshold is a genuine open research problem.** It is the target of senolytic drug discovery — drugs designed to selectively eliminate senescent cells. If AI can map the contested boundary in the literature, it may point toward drug targets that sit precisely at the tipping point.

---

## What This Project Does

```
PubMed API query (400+ abstracts)
         │
         ▼
  SciSpacy NER
  (extract GENE, PROTEIN, DISEASE, CELL_TYPE entities)
         │
         ▼
  Entity co-occurrence analysis
  (two entities in same sentence = implied relationship)
         │
         ▼
  Edge classification from evidence sentences
  ┌────────────────┬──────────────────┬──────────────────┐
  │   Activating   │    Inhibiting    │    Contested     │
  │ (activates,    │ (inhibits,       │ (papers disagree │
  │  induces,      │  suppresses,     │  about direction)│
  │  promotes)     │  blocks)         │                  │
  └────────────────┴──────────────────┴──────────────────┘
         │
         ├──────────────────────────┐
         ▼                          ▼
  NetworkX knowledge graph    BioBERT semantic clustering
  (Pyvis interactive HTML)    (abstract-level topic grouping)
         │
         ▼
  Contested edge analysis
  (the scientific finding — where experts disagree)
```

---

## Data Sources

All data is collected programmatically via the free **NCBI PubMed E-utilities API**. No manual download required — running `src/pubmed_fetch.py` retrieves everything.

| Query | Abstracts Retrieved | Focus |
|-------|:------------------:|-------|
| `autophagy AND senescence` | 200 | Core intersection |
| `mTOR AND autophagy AND insulin` | 200 | Nutrient sensing pathway |
| `AMPK AND autophagy AND aging` | 100 | Energy sensing connection |

**Total: ~400 PubMed abstracts** spanning 2010–2024.

---

## Methods

### 1. Literature Retrieval
**File:** `src/pubmed_fetch.py`

Uses [Biopython Entrez](https://biopython.org/docs/latest/api/Bio.Entrez.html) to query PubMed programmatically. Each abstract is parsed from XML format and saved with its PMID, title, abstract text, and publication year.

### 2. Named Entity Recognition
**File:** `src/extract_entities.py`

[SciSpacy](https://allenai.github.io/scispacy/) (a biomedical NLP library built on spaCy) extracts biological entities from each sentence:

| Entity Type | Examples |
|------------|---------|
| GENE / PROTEIN | mTOR, AMPK, Beclin-1, LC3, p62, ATG5 |
| DISEASE | senescence, insulin resistance, cancer |
| CELL_TYPE | fibroblast, macrophage, hepatocyte |

For each abstract, every pair of entities appearing in the **same sentence** is recorded as a potential relationship — with the full sentence as evidence.

### 3. Knowledge Graph Construction
**File:** `src/build_graph.py`

Built with [NetworkX](https://networkx.org/) as a directed graph (`DiGraph`).

**Nodes** represent biological entities with attributes:
- `entity_type` — gene, protein, process, or disease
- `mention_count` — how many abstracts mention this entity

**Edges** represent relationships with attributes:
- `weight` — number of abstracts supporting this connection
- `direction` — classified from evidence sentence keywords:

| Classification | Trigger Words | Colour |
|---------------|--------------|--------|
| Activating | activates, induces, promotes, upregulates, triggers | 🟢 Green |
| Inhibiting | inhibits, suppresses, blocks, downregulates, prevents | 🔴 Red |
| **Contested** | both directions found across different papers | 🟡 Yellow |

### 4. Visualisation
**File:** `src/visualise_graph.py`

Interactive HTML graph using [Pyvis](https://pyvis.readthedocs.io/). Open `output/graph.html` in any browser — nodes are draggable, zoomable, and hoverable to see evidence sentences.

Node size scales with `mention_count`. Edge thickness scales with `weight` (paper count).

### 5. Contested Edge Analysis
**File:** `src/contested_analysis.py`

The core scientific finding: identifies every edge where papers disagree about direction of effect. For each contested relationship, prints:
- The conflicting evidence sentences verbatim
- The PMIDs of the disagreeing papers
- The publication years (to check if consensus has shifted over time)

### 6. BioBERT Semantic Clustering
**File:** `src/bioBERT_cluster.py`

[BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2) generates 768-dimensional semantic embeddings for each abstract. KMeans clustering groups abstracts by biological topic — testing whether pro-autophagy and pro-senescence literature naturally separate at the semantic level.

---

## Key Results

### Graph Statistics

| Metric | Value |
|--------|:-----:|
| Total abstracts processed | ~400 |
| Unique biological entities | XXX |
| Total edges (relationships) | XXX |
| Activating edges | XXX |
| Inhibiting edges | XXX |
| **Contested edges** | **XXX** |
| Top hub protein (degree centrality) | mTOR |

### Top Contested Relationships

These are the most scientifically significant findings — molecular relationships where the literature contains direct contradictions:

| Protein A | Relationship | Protein B | Papers For | Papers Against |
|-----------|:-----------:|-----------|:----------:|:--------------:|
| PROTEIN_X | activates/inhibits? | PROTEIN_Y | X papers | X papers |
| PROTEIN_X | activates/inhibits? | PROTEIN_Y | X papers | X papers |
| PROTEIN_X | activates/inhibits? | PROTEIN_Y | X papers | X papers |

> These contested edges represent the current scientific uncertainty about where the autophagy-senescence threshold lies — and are candidate targets for experimental clarification or drug design.

### Interactive Knowledge Graph

![Knowledge Graph Overview](figures/graph_overview.png)

*Biological knowledge graph coloured by edge direction: green = activating, red = inhibiting, yellow = contested. Node size represents mention frequency. The yellow contested cluster sits at the boundary between the autophagy-promoting (green-dominant) and senescence-promoting (red-dominant) regions.*

### BioBERT Cluster UMAP

![BioBERT Semantic Clusters](figures/bioBERT_clusters.png)

*UMAP of BioBERT abstract embeddings. Cluster separation suggests pro-autophagy and pro-senescence literature are semantically distinct — with a boundary region containing the most contested papers.*

---

## How to Run

### Prerequisites

```bash
git clone https://github.com/YOUR-USERNAME/metabolic-ai-portfolio.git
cd metabolic-ai-portfolio/project-b-knowledge-graph
pip install -r requirements.txt
python -m spacy download en_core_sci_sm
```

You will also need a **free NCBI API key**: register at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/), go to Settings → API Key, copy it.

Create a `.env` file in this directory:
```
NCBI_API_KEY=your_key_here
NCBI_EMAIL=your@email.com
```

### Run the full pipeline

```bash
# Step 1: Fetch abstracts from PubMed
python src/pubmed_fetch.py

# Step 2: Extract biological entities with SciSpacy
python src/extract_entities.py

# Step 3: Build the knowledge graph
python src/build_graph.py

# Step 4: Generate interactive visualisation
python src/visualise_graph.py
# → opens output/graph.html in your browser

# Step 5: Analyse contested edges (the key finding)
python src/contested_analysis.py

# Step 6: BioBERT semantic clustering (optional — requires GPU or patience)
python src/bioBERT_cluster.py
```

### Run tests

```bash
pytest tests/ -v
```

---

## File Structure

```
project-b-knowledge-graph/
│
├── src/
│   ├── pubmed_fetch.py       # PubMed API retrieval via Biopython Entrez
│   ├── extract_entities.py   # SciSpacy NER on all abstracts
│   ├── build_graph.py        # NetworkX DiGraph construction + edge classification
│   ├── visualise_graph.py    # Pyvis interactive HTML graph
│   ├── contested_analysis.py # Find and report contested molecular relationships
│   └── bioBERT_cluster.py   # BioBERT embeddings + KMeans + UMAP clustering
│
├── tests/
│   ├── test_fetch.py         # Tests for PubMed retrieval
│   └── test_graph.py         # Tests for graph construction and edge classification
│
├── figures/
│   ├── graph_overview.png    # Static screenshot of knowledge graph
│   └── bioBERT_clusters.png  # UMAP of abstract semantic clusters
│
├── output/
│   └── graph.html            # Interactive Pyvis graph (open in browser)
│
├── data/
│   ├── abstracts_autophagy.json
│   ├── abstracts_mtor.json
│   └── entity_pairs.json
│
├── .env.example              # Template for your NCBI API key
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dependencies

```
biopython>=1.81
spacy>=3.6
scispacy>=0.5
networkx>=3.1
pyvis>=0.3
transformers>=4.30
torch>=2.0
umap-learn>=0.5
scikit-learn>=1.3
python-dotenv>=1.0
pytest>=7.4
```

---

## Research Context

This project addresses the pathway-level scale of the hyperinsulinaemia hypothesis:

- Insulin chronically suppresses autophagy via mTOR activation
- Impaired autophagy accelerates cellular damage accumulation
- That damage pushes cells toward senescence
- Senescent cells secrete SASP — amplifying insulin resistance systemically

The knowledge graph maps this cascade and identifies where the literature is most uncertain — which is precisely where the most impactful drug targets are likely to sit.

- **Project A** demonstrates convergence at the transcriptomic level
- **Project C** models state transitions at the physiological level

---

## References

- Choi, A.M. et al. (2013). Autophagy in human health and disease. *NEJM*, 368(7), 651–662.
- López-Otín, C. et al. (2023). Hallmarks of aging: An expanding universe. *Cell*, 186(2), 243–278.
- Lee, J. et al. (2020). BioBERT: a pre-trained biomedical language representation model. *Bioinformatics*, 36(4), 1234–1240.
- Laplante, M. & Sabatini, D.M. (2012). mTOR signaling in growth control and disease. *Cell*, 149(2), 274–293.

---

# 🔬 Project B — Autophagy-Senescence Knowledge Graph

> **Where does healthy cellular recycling transition into pathological senescent survival?**
>
> An NLP and graph analytics pipeline mining PubMed abstracts on the autophagy–cellular senescence axis. The system parses biomedical text, standardizes biological synonyms, extracts sentence-level co-occurrences with character span scoping, and classifies consensus versus contested molecular relationships across independent publications.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5?style=flat-square&logo=spacy&logoColor=white)](https://spacy.io)
[![SciSpacy](https://img.shields.io/badge/SciSpacy-0.5.4-FF6F61?style=flat-square)](https://allenai.github.io/scispacy/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.1-4B8BBE?style=flat-square)](https://networkx.org)
[![Pyvis](https://img.shields.io/badge/Pyvis-0.3.2-4ade80?style=flat-square)](https://pyvis.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

---

## The Biological Context

* **Autophagy** is the primary catabolic mechanism for degrading and recycling damaged organelles and long-lived proteins to maintain metabolic homeostasis.
* **Cellular Senescence** is a permanent cell-cycle arrest triggered by sublethal cellular stress, characterized by the secretion of pro-inflammatory cytokines termed the **SASP** (Senescence-Associated Secretory Phenotype).
* **The Literature Conflict:** Autophagy can act as a barrier against senescence by clearing damaged mitochondria, but it is also reported to be required for establishing senescence and sustaining metabolic viability in senescent states.

This project extracts these relationships directly from peer-reviewed literature to map the network structure and identify specific disputed interactions.

---

---

## Methods & Implementation

### 1. Literature Ingestion & Tag Stripping
* **Script:** `src/pubmed_fetch.py`
* Programmatically retrieves abstracts via NCBI Entrez E-Utilities.
* Regex filters strip embedded XML/HTML formatting tags (`<sup>`, `<i>`, `<b>`) before sentence segmentation to prevent broken sentence boundaries.

### 2. Entity Extraction & Normalization
* **Script:** `src/extract_entities.py`
* Uses SciSpacy's `en_core_sci_sm` biomedical language model.
* Filters non-biological noise (`review`, `human`, `target`, `damaged`, `western blotting`) and canonicalizes synonyms (`cellular senescence` → `senescence`, `senescence-associated secretory phenotype` → `sasp`).

### 3. Scoped Edge Classification
* **Script:** `src/build_graph.py`
* **Between-Span Scoping:** Action verbs are evaluated strictly within the text substring separating the two entities, eliminating full-sentence crosstalk.
* **Negation Handling:** Evaluates preceding negation phrases (`does not`, `failed to`, `without`) to prevent false-positive directional assignments.
* **Distinct PMID Contestation:** An edge is classified as `CONTESTED` only when distinct publications exist exclusively on opposing sides:
  $$\text{len}(\text{act\_pmids} - \text{inh\_pmids}) > 0 \quad \text{and} \quad \text{len}(\text{inh\_pmids} - \text{act\_pmids}) > 0$$

### 4. Interactive Physics Visualisation
* **Script:** `src/visualise_graph.py`
* Generates an interactive ForceAtlas2 network map via PyVis. Edges are color-coded by evidence classification (🟢 Activating, 🔴 Inhibiting, 🟠 Contested) with node sizes weighted by degree centrality.

### 5. Cross-Project Molecular Alignment
* **Script:** `src/cross_project_connect.py`
* Tests un-manipulated set intersection between the top 200 tissue-concordant genes from Project A and the literature-derived knowledge graph.

---

## Key Results & Graph Statistics

### High-Confidence Graph Metrics ($\ge 3$ Papers)

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Abstracts Processed** | 295 | Queried via `autophagy AND senescence` |
| **Filtered Entity Pairs** | 57,079 | Clean sentence-level co-occurrences |
| **Graph Nodes (Entities)** | 456 | Biological entities with $\ge 3$ paper support |
| **Graph Edges (Interactions)**| 745 | High-confidence interaction links |
| **Activating Edges** | 201 | Consensus activating direction |
| **Inhibiting Edges** | 122 | Consensus inhibiting direction |
| **CONTESTED Edges** | **119** | **Strict exclusive cross-paper disputes** |
| **Neutral / Nuanced Edges** | 303 | Co-occurrence without directional conflict |

### Top 10 Central Biological Hubs

| Rank | Biological Entity | Weighted Degree | Classification / Role |
| :---: | :--- | :---: | :--- |
| 1 | `senescence` | 1,784 | Core Query Hub |
| 2 | `autophagy` | 1,706 | Core Query Hub |
| 3 | `oxidative stress` | 80 | Cellular Stress Driver |
| 4 | `rapamycin` | 73 | Pharmacological Autophagy Inducer |
| 5 | `apoptosis` | 73 | Alternative Cell Death Pathway |
| 6 | `sasp` | 66 | Senescent Secretory Phenotype |
| 7 | `organelles` | 64 | Subcellular Degradation Targets |
| 8 | `mtor` | 63 | Master Nutrient Sensor |
| 9 | `p21` | 62 | Cell Cycle Arrest Effector |
| 10 | `p53` | 56 | Master Tumor Suppressor / Stress Sensor |

---

### Top Disputed Literature Interactions

Exemplar contradictory claims extracted directly from PubMed literature:

**1. `SENESCENCE` $\leftrightarrow$ `AUTOPHAGY`** (108 Activating vs 74 Inhibiting Papers)
* **[+] Activating Claim (PMID 31144309):** *"Autophagy is well known for its disruptive effect on human diseases, and it is currently proposed to have a direct effect on triggering senescence and quiescence."*
* **[-] Inhibiting Claim (PMID 34706873):** *"Re-establishment of autophagy reversed the senescent phenotype by suppressing GATA4."*

**2. `SENESCENCE` $\leftrightarrow$ `RAPAMYCIN`** (12 Activating vs 9 Inhibiting Papers)
* **[+] Activating Claim (PMID 40702750):** *"We performed a narrative review of recent mechanistic and preclinical studies investigating... interactions between autophagy impairment and senescence; and (4) the efficacy of autophagy enhancers (e.g., rapamycin and metformin)..."*
* **[-] Inhibiting Claim (PMID 39988732):** *"MTOR-dependent autophagy induced by rapamycin and torin-1 attenuated cell senescence and decreased the expression of cyclin-dependent kinase inhibitors..."*

**3. `AUTOPHAGY` $\leftrightarrow$ `OXIDATIVE STRESS`** (11 Activating vs 7 Inhibiting Papers)
* **[+] Activating Claim (PMID 36010550):** *"Further experiments revealed that autophagy was induced by artesunate treatment due to oxidative stress and ER stress."*
* **[-] Inhibiting Claim (PMID 41690118):** *"Below a critical damage threshold, robust autophagic flux suppresses senescence initiation by maintaining mitochondrial integrity, limiting oxidative stress, and preserving proteostasis."*

---

## Stability & Validation

* **Subsampling Stability:** Evaluated on a 50% bootstrap subsample ($N=150$ abstracts).
  * Global degree rank correlation: **Spearman $\rho = 0.803$ ($p = 3.41 \times 10^{-47}$)**.
  * Top-10 Hub Jaccard overlap: **66.7%**.
  * Primary contested edge (`autophagy` $\leftrightarrow$ `senescence`) remained robustly contested.
* **Threshold Sensitivity:** Comparing thresholds ($\ge 2$ papers: 156 contested across 2,127 edges vs $\ge 3$ papers: 119 contested across 745 edges) demonstrates that contested relationships concentrate within well-evidenced hubs rather than sparse noise.

---

## Methodological Limitations

1. **Query-Driven Hub Dominance:** The central prominence of `senescence` and `autophagy` is an artifact of the PubMed query string (`"autophagy AND senescence"`), not an emergent biological discovery.
2. **Corpus Scale & Lower-Rank Instability:** While ranks 1–2 are stable, ranks 3–10 shuffle under 50% subsampling due to the moderate corpus size (295 abstracts).
3. **Cross-Project Scale Mismatch:** Strict intersection between Project A's top 200 concordant genes and Project B's graph yielded 0 direct matches ($N=0$). This reflects a vocabulary resolution difference: clinical microarrays measure specific downstream transcriptomic effectors (`CNN1`, `FMO1`, `HAS2`), whereas abstract-level literature predominantly describes master regulatory complexes (`mTOR`, `SASP`, `p53`).
4. **Relational Scope:** Keyword and span-based scoping serves as a fast relational heuristic; dependency parsing would be required for strict subject-verb-object syntactical attribution.

---

## Repository Structure
project-b-knowledge-graph/
│
├── src/
│   ├── pubmed_fetch.py              # Entrez API fetcher
│   ├── extract_entities.py          # SciSpacy NER + vocabulary cleaner
│   ├── build_graph.py               # NetworkX graph builder + scoped classifier
│   ├── contested_analysis.py        # Top disputes & citation extraction
│   ├── cross_project_connect.py     # Non-circular Project A/B intersection
│   ├── evaluate_graph_robustness.py # 50% subsampling stability evaluation
│   └── visualise_graph.py           # PyVis interactive HTML renderer
│
├── tests/
│   └── test_project_b.py            # Unit tests (NER, scoping, graph logic)
│
├── figures/
│   └── knowledge_graph.html         # Standalone interactive network visual
│
├── data/                            # Raw data (.gitignore managed)
│   ├── pubmed_autophagy_senescence.json
│   ├── extracted_entity_pairs.json
│   └── cross_project_convergent_targets.csv
│
└── README.md

---

## Execution Guide

```bash
# 1. Fetch PubMed abstracts
py project-b-knowledge-graph/src/pubmed_fetch.py

# 2. Extract entities and sentence co-occurrences
py project-b-knowledge-graph/src/extract_entities.py

# 3. Construct knowledge graph & classify edges
py project-b-knowledge-graph/src/build_graph.py

# 4. Extract contested literature interactions
py project-b-knowledge-graph/src/contested_analysis.py

# 5. Evaluate graph stability (50% subsample)
py project-b-knowledge-graph/src/evaluate_graph_robustness.py

# 6. Run non-circular cross-project alignment
py project-b-knowledge-graph/src/cross_project_connect.py

# 7. Generate interactive visualization
py project-b-knowledge-graph/src/visualise_graph.py

# 8. Run unit test suite
pytest project-b-knowledge-graph/tests/test_project_b.py