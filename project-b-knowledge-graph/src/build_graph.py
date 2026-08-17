import json
import os
import re
from collections import defaultdict
import networkx as nx

# Central Synonym Mapping (Single Source of Truth)
SYNONYM_MAP = {
    "cellular senescence": "senescence",
    "senescent cells": "senescence",
    "senescent": "senescence",
    "cell senescence": "senescence",
    "macroautophagy": "autophagy",
    "autophagic flux": "autophagy",
    "autophagic": "autophagy",
    "senescence-associated secretory phenotype": "sasp"
}

ACTIVATING_KEYWORDS = [
    r"\bactivat\w*", r"\bpromot\w*", r"\binduc\w*", r"\bincreas\w*",
    r"\bstimulat\w*", r"\benhanc\w*", r"\bupregulat\w*", r"\btrigger\w*"
]
INHIBITING_KEYWORDS = [
    r"\binhibit\w*", r"\bsuppress\w*", r"\bblock\w*", r"\bprevent\w*",
    r"\bdecreas\w*", r"\battenuat\w*", r"\bdownregulat\w*", r"\breduc\w*", r"\brevers\w*"
]

NEGATION_PATTERN = re.compile(r"\b(does not|failed to|did not|unable to|without|no significant)\b", re.IGNORECASE)
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
    Returns: 'ACTIVATING', 'INHIBITING', 'MIXED', or 'NEUTRAL'
    """
    first_end = min(span_a[1], span_b[1])
    second_start = max(span_a[0], span_b[0])
    
    if first_end < second_start:
        in_between_text = sentence[first_end:second_start]
    else:
        start_win = max(0, min(span_a[0], span_b[0]) - 20)
        end_win = min(len(sentence), max(span_a[1], span_b[1]) + 20)
        in_between_text = sentence[start_win:end_win]
        
    has_act = bool(ACT_PATTERN.search(in_between_text))
    has_inh = bool(INH_PATTERN.search(in_between_text))
    is_negated = bool(NEGATION_PATTERN.search(in_between_text))
    
    if is_negated:
        return "NEUTRAL"
        
    if has_act and has_inh:
        return "MIXED"
    elif has_act:
        return "ACTIVATING"
    elif has_inh:
        return "INHIBITING"
        
    return "NEUTRAL"


def build_knowledge_graph(min_paper_evidence=3):
    """
    Constructs a NetworkX graph requiring strict exclusive cross-paper disagreement
    (papers exclusively on activating side AND papers exclusively on inhibiting side).
    """
    print("\n==================================================")
    print(f"=== BUILDING KNOWLEDGE GRAPH (THRESHOLD >= {min_paper_evidence} PAPERS) ===")
    print("==================================================")
    
    input_pairs_path = "project-b-knowledge-graph/data/extracted_entity_pairs.json"
    with open(input_pairs_path, "r", encoding="utf-8") as f:
        pair_records = json.load(f)
        
    print(f"Loaded {len(pair_records)} filtered co-occurrence records.")
    
    pair_evidence = defaultdict(lambda: {
        "all_pmids": set(),
        "act_pmids": set(),
        "inh_pmids": set(),
        "mixed_pmids": set(),
        "sentences": []
    })
    
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
        
        rel = classify_scoped_relationship(sent, span_a, span_b)
        
        pair_evidence[pair_key]["all_pmids"].add(pmid)
        pair_evidence[pair_key]["sentences"].append(sent)
        
        if rel == "ACTIVATING":
            pair_evidence[pair_key]["act_pmids"].add(pmid)
        elif rel == "INHIBITING":
            pair_evidence[pair_key]["inh_pmids"].add(pmid)
        elif rel == "MIXED":
            pair_evidence[pair_key]["mixed_pmids"].add(pmid)

    G = nx.Graph()
    contested_count = 0
    activating_count = 0
    inhibiting_count = 0
    neutral_count = 0
    
    for (node_u, node_v), data in pair_evidence.items():
        total_papers = len(data["all_pmids"])
        
        if total_papers < min_paper_evidence:
            continue
            
        act_p = data["act_pmids"]
        inh_p = data["inh_pmids"]
        
        # Strict cross-paper dispute: at least one exclusive paper on each side
        has_exclusive_act = len(act_p - inh_p) > 0
        has_exclusive_inh = len(inh_p - act_p) > 0
        is_strict_disagreement = has_exclusive_act and has_exclusive_inh
        
        if is_strict_disagreement:
            classification = "CONTESTED"
            contested_count += 1
        elif len(act_p) > 0 and len(inh_p) == 0:
            classification = "ACTIVATING"
            activating_count += 1
        elif len(inh_p) > 0 and len(act_p) == 0:
            classification = "INHIBITING"
            inhibiting_count += 1
        else:
            classification = "NEUTRAL"
            neutral_count += 1
            
        G.add_edge(
            node_u,
            node_v,
            weight=total_papers,
            classification=classification,
            activating_papers=len(act_p),
            inhibiting_papers=len(inh_p),
            mixed_papers=len(data["mixed_pmids"]),
            pmid_count=total_papers,
            evidence_sample=data["sentences"][0]
        )

    print(f"\n[Graph Summary at Threshold >= {min_paper_evidence} Papers]")
    print(f"  Nodes (Biological Entities): {G.number_of_nodes()}")
    print(f"  Edges (High-Confidence Links): {G.number_of_edges()}")
    
    print(f"\n[Edge Classification Breakdown]")
    print(f"  Activating Edges: {activating_count}")
    print(f"  Inhibiting Edges: {inhibiting_count}")
    print(f"  CONTESTED Edges:  {contested_count} (Strict Cross-Paper Disputes)")
    print(f"  Neutral / Nuanced: {neutral_count}")

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