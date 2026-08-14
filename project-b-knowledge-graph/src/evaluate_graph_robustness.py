import json
import random
from collections import defaultdict
import networkx as nx
from scipy.stats import spearmanr
from build_graph import canonicalize_entity, classify_sentence_relationship


def run_subsample_graph_pipeline(pair_records, pmid_subset, min_paper_evidence=2):
    """Rebuilds a knowledge graph using only a subset of PMIDs."""
    pair_evidence = defaultdict(lambda: {"pmids": set(), "act_count": 0, "inh_count": 0})
    
    for rec in pair_records:
        # Keep only pairs originating from the subsampled PMIDs
        if rec["pmid"] not in pmid_subset:
            continue
            
        ent_a = canonicalize_entity(rec["entity_a"])
        ent_b = canonicalize_entity(rec["entity_b"])
        if not ent_a or not ent_b or ent_a == ent_b:
            continue
            
        pair_key = tuple(sorted([ent_a, ent_b]))
        rel = classify_sentence_relationship(rec["evidence_sentence"])
        
        if rel == "ACTIVATING":
            pair_evidence[pair_key]["act_count"] += 1
        elif rel == "INHIBITING":
            pair_evidence[pair_key]["inh_count"] += 1
            
        pair_evidence[pair_key]["pmids"].add(rec["pmid"])

    G_sub = nx.Graph()
    for (node_u, node_v), data in pair_evidence.items():
        paper_count = len(data["pmids"])
        if paper_count < min_paper_evidence:
            continue
            
        acts = data["act_count"]
        inhs = data["inh_count"]
        
        if acts > 0 and inhs > 0:
            classification = "CONTESTED"
        elif acts > 0:
            classification = "ACTIVATING"
        elif inhs > 0:
            classification = "INHIBITING"
        else:
            classification = "NEUTRAL"
            
        G_sub.add_edge(
            node_u,
            node_v,
            weight=paper_count,
            classification=classification,
            activating_signals=acts,
            inhibiting_signals=inhs,
            pmid_count=paper_count
        )
        
    return G_sub


def evaluate_graph_robustness():
    print("\n==================================================")
    print("=== KNOWLEDGE GRAPH STABILITY & ROBUSTNESS ===")
    print("=== (FULL N=295 vs 50% SUBSAMPLE N=150) ===")
    print("==================================================")
    
    pairs_path = "project-b-knowledge-graph/data/extracted_entity_pairs.json"
    with open(pairs_path, "r", encoding="utf-8") as f:
        all_pairs = json.load(f)
        
    all_pmids = list(set(rec["pmid"] for rec in all_pairs))
    print(f"Total Unique PMIDs in Full Dataset: {len(all_pmids)}")
    
    # 1. Build Full Graph (N=295)
    G_full = run_subsample_graph_pipeline(all_pairs, set(all_pmids), min_paper_evidence=2)
    deg_full = dict(G_full.degree(weight="weight"))
    top10_full = [node for node, _ in sorted(deg_full.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    # 2. Build 50% Subsampled Graph (N=150)
    random.seed(42)
    subsample_pmids = set(random.sample(all_pmids, 150))
    G_half = run_subsample_graph_pipeline(all_pairs, subsample_pmids, min_paper_evidence=2)
    deg_half = dict(G_half.degree(weight="weight"))
    top10_half = [node for node, _ in sorted(deg_half.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    # 3. Calculate Stability Metrics
    jaccard_hubs = len(set(top10_full) & set(top10_half)) / len(set(top10_full) | set(top10_half))
    
    # Rank correlation across shared nodes
    shared_nodes = list(set(G_full.nodes()) & set(G_half.nodes()))
    ranks_full = [deg_full[n] for n in shared_nodes]
    ranks_half = [deg_half[n] for n in shared_nodes]
    rho, p_val = spearmanr(ranks_full, ranks_half)
    
    print("\n--- TOP 10 HUBS COMPARISON ---")
    print(f"{'Rank':<5} | {'Full Dataset (N=295)':<25} | {'50% Subsample (N=150)':<25}")
    print("-" * 60)
    for r in range(10):
        f_node = top10_full[r] if r < len(top10_full) else "N/A"
        h_node = top10_half[r] if r < len(top10_half) else "N/A"
        print(f"{r+1:<5} | {f_node:<25} | {h_node:<25}")
        
    print("\n--- ROBUSTNESS SUMMARY METRICS ---")
    print(f"Top 10 Hub Jaccard Overlap:        {jaccard_hubs*100:.1f}%")
    print(f"Global Degree Spearman Rank Rho:   {rho:.3f} (p = {p_val:.4e})")
    
    # 4. Check if top contested edge survived
    top_contested_survived = G_half.has_edge("autophagy", "senescence") and \
                             G_half["autophagy"]["senescence"]["classification"] == "CONTESTED"
    print(f"Primary Contested Edge ('autophagy' <-> 'senescence') Retained: {top_contested_survived}")


if __name__ == "__main__":
    evaluate_graph_robustness()