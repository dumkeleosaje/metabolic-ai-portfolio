import json
import os
import networkx as nx
from build_graph import build_knowledge_graph, classify_sentence_relationship, canonicalize_entity


def run_contested_edge_analysis(top_n=5):
    """
    Identifies the top contested molecular relationships in the knowledge graph
    and prints their conflicting evidence sentences side-by-side.
    """
    print("\n==================================================")
    print("=== TOP CONTESTED LITERATURE INTERACTIONS ===")
    print("==================================================")
    
    # 1. Build the graph and get raw co-occurrence data
    G = build_knowledge_graph(min_paper_evidence=2)
    
    input_pairs_path = "project-b-knowledge-graph/data/extracted_entity_pairs.json"
    with open(input_pairs_path, "r", encoding="utf-8") as f:
        pair_records = json.load(f)
        
    # Index all evidence sentences by pair
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
        rel = classify_sentence_relationship(sent)
        
        if rel in ["ACTIVATING", "INHIBITING"]:
            pair_evidence_sentences[pair_key][rel].append({"pmid": pmid, "sentence": sent})

    # 2. Extract and rank contested edges
    contested_edges = []
    for u, v, data in G.edges(data=True):
        if data.get("classification") == "CONTESTED":
            acts = data["activating_signals"]
            inhs = data["inhibiting_signals"]
            # Contention score: product of opposing evidence signals
            score = acts * inhs
            contested_edges.append({
                "pair": (u, v),
                "acts": acts,
                "inhs": inhs,
                "score": score,
                "total_papers": data["pmid_count"]
            })
            
    contested_edges = sorted(contested_edges, key=lambda x: x["score"], reverse=True)
    
    print(f"\nDiscovered {len(contested_edges)} contested relationships. Displaying Top {top_n} Disputes:\n")
    
    # 3. Print Side-by-Side Conflicting Citations
    for rank, item in enumerate(contested_edges[:top_n], 1):
        u, v = item["pair"]
        pair_key = tuple(sorted([u, v]))
        evidence = pair_evidence_sentences.get(pair_key, {})
        
        print("----------------------------------------------------------------------")
        print(f"[{rank}] DISPUTED RELATIONSHIP: '{u.upper()}' <---> '{v.upper()}'")
        print(f"    Evidence Balance: {item['acts']} Activating vs {item['inhs']} Inhibiting ({item['total_papers']} Papers)")
        print("----------------------------------------------------------------------")
        
        # Activating Evidence
        if evidence.get("ACTIVATING"):
            sample_act = evidence["ACTIVATING"][0]
            print(f"  [+] ACTIVATING CLAIM (PMID: {sample_act['pmid']}):")
            print(f"      \"{sample_act['sentence']}\"\n")
            
        # Inhibiting Evidence
        if evidence.get("INHIBITING"):
            sample_inh = evidence["INHIBITING"][0]
            print(f"  [-] INHIBITING CLAIM (PMID: {sample_inh['pmid']}):")
            print(f"      \"{sample_inh['sentence']}\"\n")


if __name__ == "__main__":
    run_contested_edge_analysis(top_n=5)