# Project B: Autophagy-Senescence Knowledge Graph

Where does the research literature on autophagy and cellular senescence disagree with itself? I mined 295 PubMed abstracts, built a knowledge graph of the relationships they describe, and flagged the ones where different papers claim opposite effects.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5?style=flat-square&logo=spacy&logoColor=white)](https://spacy.io)
[![SciSpacy](https://img.shields.io/badge/SciSpacy-0.5.4-FF6F61?style=flat-square)](https://allenai.github.io/scispacy/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.1-4B8BBE?style=flat-square)](https://networkx.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

---

## Background

**Autophagy** is how cells break down and recycle damaged proteins and organelles. Fasting and low nutrient levels switch it on, and high insulin (through mTOR) switches it off.

**Cellular senescence** is a permanent stop to cell division after damage. Senescent cells don't die. They stay around and release inflammatory signals (the SASP), and they build up with age and in metabolic disease.

The two are linked, but not in a simple way. Some papers describe autophagy as protective, clearing damage before it can push a cell into senescence. Others describe autophagy as something senescent cells need in order to survive. I wanted to find exactly where in the literature those claims collide.

---

## How it works

1. **Fetch.** Pull abstracts for `autophagy AND senescence` from PubMed with Biopython's Entrez API. 300 PMIDs came back, and 295 had usable text.
2. **Clean.** Strip embedded HTML/XML tags (`<sup>`, `<i>`) *before* splitting into sentences. Leaving them in breaks spaCy's sentence splitter.
3. **Extract entities.** Run SciSpacy (`en_core_sci_sm`) on each sentence and keep each entity's character position.
4. **Filter.** Remove around 130 non-biological terms (words like "review", "human", "western blotting") and merge synonyms, for example "cellular senescence" into "senescence". The synonym map lives in one place (`build_graph.py`) and the extractor imports it.
5. **Classify.** For each pair of entities in a sentence, look only at the verbs *between* the two entities. In "autophagy reversed the senescent phenotype by suppressing GATA4", the word "suppressing" refers to GATA4, not to the autophagy-senescence relationship, so scanning the whole sentence would get it wrong. A negation ("does not inhibit") counts as neutral rather than flipping the direction.
6. **Build the graph.** NetworkX, keeping only relationships supported by at least 3 papers.
7. **Find contested edges.** An edge counts as contested only if at least one paper is *only* on the activating side and at least one is *only* on the inhibiting side. That separates real disagreement between papers from a single paper describing both effects.
8. **Visualise.** An interactive Pyvis graph of the top hubs.

---

## Results

| | Count |
|---|:---:|
| Abstracts | 295 |
| Sentence-level entity pairs | 57,079 |
| Entities in graph (≥3 papers) | 456 |
| Relationships in graph | 745 |
| Activating | 201 |
| Inhibiting | 122 |
| **Contested** | **119** |
| Neutral or mixed | 303 |

The contested count began at **402** and fell to **119** over four rounds of fixing my own method. Each round removed a different kind of artefact: whole-sentence verb matching (402), double-counting mixed sentences (258), counting papers rather than sentences (127), and finally requiring papers to be exclusively on one side (119). Getting that number right taught me more than the number itself.

**Top hubs by weighted degree:** senescence (1,784), autophagy (1,706), oxidative stress (80), rapamycin (73), apoptosis (73), SASP (66), organelles (64), mTOR (63), p21 (62), p53 (56).

### A genuine disagreement

**Autophagy and senescence** (108 activating vs 74 inhibiting papers):
- *For:* "...it is currently proposed to have a direct effect on triggering senescence and quiescence." (PMID 31144309)
- *Against:* "Re-establishment of autophagy reversed the senescent phenotype by suppressing GATA4." (PMID 34706873)

### Where the method gets it wrong

Not every "contested" edge is a real disagreement, and looking at the examples shows why:
- **Autophagy and oxidative stress.** "Autophagy was induced by... oxidative stress" (PMID 36010550) and "autophagic flux... limiting oxidative stress" (PMID 41690118) were flagged as contradictory. They describe opposite *directions* (stress triggers autophagy, and autophagy reduces stress), which is a feedback loop rather than a disagreement. The graph doesn't track which entity acts on which, so it can't tell these apart.
- **Rapamycin and senescence.** One of the "activating" sentences comes from a review that simply lists rapamycin as an autophagy enhancer. Keyword classification picked up a relationship the sentence doesn't actually claim.

These are the main reasons I'd move to dependency parsing next.

---

## Robustness

- **Half-sample test** (150 abstracts, same ≥3-paper threshold): degree rank correlation Spearman ρ = 0.80, top-10 hub overlap 66.7%, and the autophagy-senescence edge stayed contested. The top two hubs are stable. Ranks 3 to 10 shuffle noticeably, which is expected with a small corpus.
- **Threshold sensitivity:** lowering the threshold to ≥2 papers nearly triples the graph (745 to 2,127 edges) but only raises contested edges from 119 to about 156. Disagreements cluster in well-studied relationships rather than appearing as noise.

---

## Link to Project A

I checked whether Project A's top 200 muscle-liver concordant genes appear in this graph. None do. That test is weak by design: the literature graph is dominated by broad processes (oxidative stress, apoptosis), while Project A produces individual genes (NR4A3, FMO1, CNN1). It's a vocabulary mismatch more than a biological result. An earlier version looked more positive, but only because I'd mixed a hand-picked list of "metabolic genes" into the comparison. I removed that because it was circular.

---

## Limitations

- The top two hubs being senescence and autophagy is guaranteed by the search query. It isn't a finding.
- Two entities in the same sentence doesn't mean they interact.
- Keyword classification is a heuristic, and relationships aren't directed (see above).
- Abstracts only, not full texts.
- SciSpacy entity errors carry through uncorrected.
- 295 abstracts is a small corpus.

---

## Running it

```bash
git clone https://github.com/dumkeleosaje/metabolic-ai-portfolio.git
cd metabolic-ai-portfolio/project-b-knowledge-graph
pip install -r requirements.txt
python -m spacy download en_core_sci_sm

python src/pubmed_fetch.py               # fetch abstracts from PubMed
python src/extract_entities.py           # entities and sentence co-occurrences
python src/build_graph.py                # build the graph and classify edges
python src/contested_analysis.py         # list contested relationships with evidence
python src/evaluate_graph_robustness.py  # half-sample stability check
python src/cross_project_connect.py      # overlap with Project A genes
python src/visualise_graph.py            # interactive graph -> figures/knowledge_graph.html

pytest tests/test_project_b.py
```

Open `figures/knowledge_graph.html` in a browser to explore the graph.

```
project-b-knowledge-graph/
├── src/        # pipeline scripts (above)
├── tests/      # unit tests for NER cleaning, span scoping and graph logic
├── figures/    # interactive graph
├── data/       # fetched abstracts and extracted pairs (not committed)
└── README.md
```

---

## References

- Choi, A.M. et al. (2013). Autophagy in human health and disease. *NEJM*, 368(7), 651-662.
- López-Otín, C. et al. (2023). Hallmarks of aging: an expanding universe. *Cell*, 186(2), 243-278.
- Laplante, M. & Sabatini, D.M. (2012). mTOR signaling in growth control and disease. *Cell*, 149(2), 274-293.
