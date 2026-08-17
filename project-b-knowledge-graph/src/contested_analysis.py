import json
import os
import networkx as nx
from build_graph import (
    build_knowledge_graph,
    classify_scoped_relationship,
    canonicalize_entity
)


def run_contested_edge_analysis(top_n=5, min_paper_evidence=3):
    """
    Identifies the top contested molecular relationships in the knowledge graph
    using between-span verb scoping and prints conflicting evidence sentences.
    """
    print("\n==================================================")
    print("=== TOP CONTESTED LITERATURE INTERACTIONS ===")
    print("=== (SCOPED VERB CLASSIFICATION, >= 3 PAPERS) ===")
    print("==================================================")
    
    # 1. Build the graph with min_paper_evidence=3
    G = build_knowledge_graph(min_paper_evidence=min_paper_evidence)
    
    input_pairs_path = "project-b-knowledge-graph/data/extracted_entity_pairs.json"
    with open(input_pairs_path, "r", encoding="utf-8") as f:
        pair_records = json.load(f)
        
    # Index scoped evidence sentences by pair
    pair_evidence_sentences = {}
    for rec in pair_records:
        ent_a = canonicalize_entity(rec["entity_a"])
        ent_b = canonicalize_entity(rec["entity_b"])
        if not ent_a or not ent_b or ent_a == ent_b:
            continue
        
        pair_key = tuple(sorted([ent_a, ent_b]))
        if pair_key not in pair_evidence_sentences:
            pair_evidence_sentences[pair_key] = {"ACTIVATING": [], "INHIBITING": []}
            
        sent = rec["evidence_sentence"]
        pmid = rec["pmid"]
        span_a = rec.get("span_a", [0, len(ent_a)])
        span_b = rec.get("span_b", [0, len(ent_b)])
        
        # Scoped classification
        rel = classify_scoped_relationship(sent, span_a, span_b)
        
        if rel in ["ACTIVATING", "INHIBITING"]:
            pair_evidence_sentences[pair_key][rel].append({"pmid": pmid, "sentence": sent})

    # 2. Extract and rank contested edges by contention score (acts * inhs)
    contested_edges = []
    for u, v, data in G.edges(data=True):
        if data.get("classification") == "CONTESTED":
            acts = data["activating_signals"]
            inhs = data["inhibiting_signals"]
            score = acts * inhs
            contested_edges.append({
                "pair": (u, v),
                "acts": acts,
                "inhs": inhs,
                "score": score,
                "total_papers": data["pmid_count"]
            })
            
    contested_edges = sorted(contested_edges, key=lambda x: x["score"], reverse=True)
    
    print(f"\nDiscovered {len(contested_edges)} Defensible Contested Relationships. Top {top_n} Disputes:\n")
    
    # 3. Print Side-by-Side Conflicting Citations
    for rank, item in enumerate(contested_edges[:top_n], 1):
        u, v = item["pair"]
        pair_key = tuple(sorted([u, v]))
        evidence = pair_evidence_sentences.get(pair_key, {})
        
        print("----------------------------------------------------------------------")
        print(f"[{rank}] DISPUTED RELATIONSHIP: '{u.upper()}' <---> '{v.upper()}'")
        print(f"    Evidence Balance: {item['acts']} Activating vs {item['inhs']} Inhibiting ({item['total_papers']} Papers)")
        print("----------------------------------------------------------------------")
        
        if evidence.get("ACTIVATING"):
            sample_act = evidence["ACTIVATING"][0]
            print(f"  [+] ACTIVATING CLAIM (PMID: {sample_act['pmid']}):")
            print(f"      \"{sample_act['sentence']}\"\n")
            
        if evidence.get("INHIBITING"):
            sample_inh = evidence["INHIBITING"][0]
            print(f"  [-] INHIBITING CLAIM (PMID: {sample_inh['pmid']}):")
            print(f"      \"{sample_inh['sentence']}\"\n")


if __name__ == "__main__":
    run_contested_edge_analysis(top_n=5, min_paper_evidence=3)