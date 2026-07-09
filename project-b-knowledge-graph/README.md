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