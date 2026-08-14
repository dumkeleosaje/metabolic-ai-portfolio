import json
import os
import re
from collections import defaultdict
import networkx as nx

# Action words to exclude from node vocabulary so they don't become false entities
ACTION_WORDS_BLACKLIST = {
    "increased", "decreased", "inhibition", "activation", "induced", "treatment",
    "accumulation", "associated with", "development", "expression", "suppression",
    "induction", "regulation", "mediating", "enhanced", "attenuated", "reduced"
}

# Synonym mapping to group identical biological concepts into clean graph nodes
SYNONYM_MAP = {
    "cellular senescence": "senescence",
    "senescent cells": "senescence",
    "senescent": "senescence",
    "macroautophagy": "autophagy",
    "autophagic flux": "autophagy",
    "autophagic": "autophagy"
}

# Regex patterns for relationship classification
ACTIVATING_KEYWORDS = [
    r"\bactivat\w*", r"\bpromot\w*", r"\binduc\w*", r"\bincreas\w*",
    r"\bstimulat\w*", r"\benhanc\w*", r"\bupregulat\w*", r"\btrigger\w*"
]
INHIBITING_KEYWORDS = [
    r"\binhibit\w*", r"\bsuppress\w*", r"\bblock\w*", r"\bprevent\w*",
    r"\bdecreas\w*", r"\battenuat\w*", r"\bdownregulat\w*", r"\breduc\w*"
]

ACT_PATTERN = re.compile("|".join(ACTIVATING_KEYWORDS), re.IGNORECASE)
INH_PATTERN = re.compile("|".join(INHIBITING_KEYWORDS), re.IGNORECASE)


def canonicalize_entity(ent):
    """Maps entity to lower-case canonical form or rejects if it is in the blacklist."""
    ent = ent.strip().lower()
    if ent in ACTION_WORDS_BLACKLIST or len(ent) <= 2:
        return None
    return SYNONYM_MAP.get(ent, ent)


def classify_sentence_relationship(sentence):
    """
    Scans a single evidence sentence for activating vs inhibiting verbs.
    Returns: 'ACTIVATING', 'INHIBITING', or 'NEUTRAL'
    """
    has_act = bool(ACT_PATTERN.search(sentence))
    has_inh = bool(INH_PATTERN.search(sentence))
    
    if has_act and not has_inh:
        return "ACTIVATING"
    elif has_inh and not has_act:
        return "INHIBITING"
    elif has_act and has_inh:
        return "MIXED_SENTENCE"
    return "NEUTRAL"


def build_knowledge_graph(min_paper_evidence=2):
    """
    Aggregates co-occurrences into a NetworkX graph and classifies contested edges.
    """
    print("\n==================================================")
    print("=== BUILDING KNOWLEDGE GRAPH & EDGE CLASSIFIER ===")
    print("==================================================")
    
    input_pairs_path = "project-b-knowledge-graph/data/extracted_entity_pairs.json"
    with open(input_pairs_path, "r", encoding="utf-8") as f:
        pair_records = json.load(f)
        
    print(f"Loaded {len(pair_records)} raw co-occurrence records.")
    
    # Group evidence sentences and PMIDs by normalized entity pair
    pair_evidence = defaultdict(lambda: {"sentences": [], "pmids": set(), "act_count": 0, "inh_count": 0})
    
    for rec in pair_records:
        ent_a = canonicalize_entity(rec["entity_a"])
        ent_b = canonicalize_entity(rec["entity_b"])
        
        if not ent_a or not ent_b or ent_a == ent_b:
            continue
            
        # Standardize node ordering (undirected key)
        pair_key = tuple(sorted([ent_a, ent_b]))
        sent = rec["evidence_sentence"]
        pmid = rec["pmid"]
        
        rel = classify_sentence_relationship(sent)
        if rel == "ACTIVATING":
            pair_evidence[pair_key]["act_count"] += 1
        elif rel == "INHIBITING":
            pair_evidence[pair_key]["inh_count"] += 1
            
        pair_evidence[pair_key]["sentences"].append(sent)
        pair_evidence[pair_key]["pmids"].add(pmid)

    # Construct the NetworkX Graph
    G = nx.Graph()
    contested_count = 0
    activating_count = 0
    inhibiting_count = 0
    neutral_count = 0
    
    for (node_u, node_v), data in pair_evidence.items():
        paper_count = len(data["pmids"])
        
        # Prune spurious 1-paper noise edges to keep graph robust
        if paper_count < min_paper_evidence:
            continue
            
        acts = data["act_count"]
        inhs = data["inh_count"]
        
        # Determine Edge Classification
        if acts > 0 and inhs > 0:
            classification = "CONTESTED"
            contested_count += 1
        elif acts > 0 and inhs == 0:
            classification = "ACTIVATING"
            activating_count += 1
        elif inhs > 0 and acts == 0:
            classification = "INHIBITING"
            inhibiting_count += 1
        else:
            classification = "NEUTRAL"
            neutral_count += 1
            
        G.add_edge(
            node_u,
            node_v,
            weight=paper_count,
            classification=classification,
            activating_signals=acts,
            inhibiting_signals=inhs,
            pmid_count=paper_count,
            evidence_sample=data["sentences"][0]  # Store first sentence as representative
        )

    print(f"\n[Graph Summary]")
    print(f"  Nodes (Entities): {G.number_of_nodes()}")
    print(f"  Edges (Interactions with >= {min_paper_evidence} papers): {G.number_of_edges()}")
    
    print(f"\n[Edge Classification Breakdown]")
    print(f"  Activating Edges: {activating_count}")
    print(f"  Inhibiting Edges: {inhibiting_count}")
    print(f"  CONTESTED Edges:  {contested_count} (Literature Disagreement)")
    print(f"  Neutral Edges:    {neutral_count}")

    # Top Central Hubs
    degree_dict = dict(G.degree(weight="weight"))
    sorted_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\n==================================================")
    print("=== TOP 10 CENTRAL HUBS (BY WEIGHTED DEGREE) ===")
    print("==================================================")
    for rank, (node, deg) in enumerate(sorted_hubs, 1):
        print(f"  {rank:2d}. {node:<25} (Degree Weight: {deg})")
        
    return G


if __name__ == "__main__":
    build_knowledge_graph(min_paper_evidence=2)