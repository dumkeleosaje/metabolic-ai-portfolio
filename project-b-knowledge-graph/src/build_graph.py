import json
import os
import re
from collections import defaultdict
import networkx as nx

# Expanded synonym dictionary to standardise entity naming
SYNONYM_MAP = {
    "cellular senescence": "senescence",
    "senescent cells": "senescence",
    "senescent": "senescence",
    "cell senescence": "senescence",
    "macroautophagy": "autophagy",
    "autophagic flux": "autophagy",
    "autophagic": "autophagy"
}

# Regex patterns for activating and inhibiting relationships
ACTIVATING_KEYWORDS = [
    r"\bactivat\w*", r"\bpromot\w*", r"\binduc\w*", r"\bincreas\w*",
    r"\bstimulat\w*", r"\benhanc\w*", r"\bupregulat\w*", r"\btrigger\w*"
]
INHIBITING_KEYWORDS = [
    r"\binhibit\w*", r"\bsuppress\w*", r"\bblock\w*", r"\bprevent\w*",
    r"\bdecreas\w*", r"\battenuat\w*", r"\bdownregulat\w*", r"\breduc\w*", r"\brevers\w*"
]

NEGATION_PATTERN = re.compile(r"\b(not|never|no|failed to|unable to|without)\b", re.IGNORECASE)
HEDGING_PATTERN = re.compile(r"\b(may|might|could|potentially|suggests|hypothesized)\b", re.IGNORECASE)

ACT_PATTERN = re.compile("|".join(ACTIVATING_KEYWORDS), re.IGNORECASE)
INH_PATTERN = re.compile("|".join(INHIBITING_KEYWORDS), re.IGNORECASE)


def canonicalize_entity(ent):
    """Maps entity to canonical vocabulary."""
    if not ent or len(ent) <= 2:
        return None
    ent = ent.strip().lower()
    return SYNONYM_MAP.get(ent, ent)


def classify_scoped_relationship(sentence, span_a, span_b):
    """
    Classifies relationship strictly within the text span between entity_a and entity_b.
    Prevents whole-sentence crosstalk and handles negation.
    """
    # Determine start and end of the text between the two entities
    first_end = min(span_a[1], span_b[1])
    second_start = max(span_a[0], span_b[0])
    
    # Extract the text between entities (or a window of +/- 30 chars if adjacent)
    if first_end < second_start:
        in_between_text = sentence[first_end:second_start]
    else:
        # If spans overlap or are adjacent, look at a local 40-char window
        start_win = max(0, min(span_a[0], span_b[0]) - 20)
        end_win = min(len(sentence), max(span_a[1], span_b[1]) + 20)
        in_between_text = sentence[start_win:end_win]
        
    has_act = bool(ACT_PATTERN.search(in_between_text))
    has_inh = bool(INH_PATTERN.search(in_between_text))
    is_negated = bool(NEGATION_PATTERN.search(in_between_text))
    
    # Handle negation inversions (e.g., "does not activate" -> INHIBITING/NOT_ACT)
    if is_negated:
        if has_act and not has_inh:
            return "INHIBITING"
        elif has_inh and not has_act:
            return "ACTIVATING"
            
    if has_act and not has_inh:
        return "ACTIVATING"
    elif has_inh and not has_act:
        return "INHIBITING"
    elif has_act and has_inh:
        return "MIXED"
        
    return "NEUTRAL"


def build_knowledge_graph(min_paper_evidence=3):
    """
    Builds a robust NetworkX graph using between-span verb scoping
    and a higher confidence threshold (>= 3 papers).
    """
    print("\n==================================================")
    print("=== BUILDING KNOWLEDGE GRAPH (SCOPED VERB LOGIC) ===")
    print("==================================================")
    
    input_pairs_path = "project-b-knowledge-graph/data/extracted_entity_pairs.json"
    with open(input_pairs_path, "r", encoding="utf-8") as f:
        pair_records = json.load(f)
        
    print(f"Loaded {len(pair_records)} filtered co-occurrence records.")
    
    pair_evidence = defaultdict(lambda: {"sentences": [], "pmids": set(), "act_count": 0, "inh_count": 0})
    
    for rec in pair_records:
        ent_a = canonicalize_entity(rec["entity_a"])
        ent_b = canonicalize_entity(rec["entity_b"])
        
        if not ent_a or not ent_b or ent_a == ent_b:
            continue
            
        pair_key = tuple(sorted([ent_a, ent_b]))
        sent = rec["evidence_sentence"]
        pmid = rec["pmid"]
        span_a = rec.get("span_a", [0, len(ent_a)])
        span_b = rec.get("span_b", [0, len(ent_b)])
        
        # Apply Scoped Span Classification
        rel = classify_scoped_relationship(sent, span_a, span_b)
        
        if rel == "ACTIVATING":
            pair_evidence[pair_key]["act_count"] += 1
        elif rel == "INHIBITING":
            pair_evidence[pair_key]["inh_count"] += 1
            
        pair_evidence[pair_key]["sentences"].append(sent)
        pair_evidence[pair_key]["pmids"].add(pmid)

    # Build NetworkX Graph
    G = nx.Graph()
    contested_count = 0
    activating_count = 0
    inhibiting_count = 0
    neutral_count = 0
    
    for (node_u, node_v), data in pair_evidence.items():
        paper_count = len(data["pmids"])
        
        # High confidence threshold: >= 3 papers
        if paper_count < min_paper_evidence:
            continue
            
        acts = data["act_count"]
        inhs = data["inh_count"]
        
        if acts > 0 and inhs > 0:
            classification = "CONTESTED"
            contested_count += 1
        elif acts > 0:
            classification = "ACTIVATING"
            activating_count += 1
        elif inhs > 0:
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
            evidence_sample=data["sentences"][0]
        )

    print(f"\n[Graph Summary at Threshold >= {min_paper_evidence} Papers]")
    print(f"  Nodes (Biological Entities): {G.number_of_nodes()}")
    print(f"  Edges (High-Confidence Links): {G.number_of_edges()}")
    
    print(f"\n[Edge Classification Breakdown]")
    print(f"  Activating Edges: {activating_count}")
    print(f"  Inhibiting Edges: {inhibiting_count}")
    print(f"  CONTESTED Edges:  {contested_count} (Defensible Literature Disputes)")
    print(f"  Neutral Edges:    {neutral_count}")

    # Top Central Hubs
    degree_dict = dict(G.degree(weight="weight"))
    sorted_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\n==================================================")
    print("=== TOP 10 CENTRAL BIOLOGICAL HUBS ===")
    print("==================================================")
    for rank, (node, deg) in enumerate(sorted_hubs, 1):
        print(f"  {rank:2d}. {node:<25} (Degree Weight: {deg})")
        
    return G


if __name__ == "__main__":
    build_knowledge_graph(min_paper_evidence=3)