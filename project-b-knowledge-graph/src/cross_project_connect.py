import os
import pandas as pd
import networkx as nx
from build_graph import build_knowledge_graph


def run_honest_cross_project_connection():
    print("\n==================================================")
    print("=== HONEST NON-CIRCULAR CROSS-PROJECT ALIGNMENT ===")
    print("=== (PROJECT A CONCORDANT GENES vs PROJECT B GRAPH) ===")
    print("==================================================")
    
    # 1. Build Project B Knowledge Graph (>= 3 papers)
    G = build_knowledge_graph(min_paper_evidence=3)
    graph_nodes = set(node.lower() for node in G.nodes())
    print(f"Project B Graph contains {len(graph_nodes)} unique biological entities.")
    
    # 2. Load STRICTLY the top 200 concordant genes from Project A (NO hardcoded sets!)
    proj_a_genes_path = "project-a-convergence-classifier/data/muscle_liver_concordant_genes.csv"
    
    if not os.path.exists(proj_a_genes_path):
        raise FileNotFoundError(f"Project A concordant gene file not found at: {proj_a_genes_path}")
        
    df_proj_a = pd.read_csv(proj_a_genes_path)
    proj_a_genes = set(df_proj_a["Gene_Symbol"].astype(str).str.lower().tolist())
    print(f"Loaded {len(proj_a_genes)} unbiased concordant genes from Project A.")
    
    # 3. Honest Set Intersection (Strict A ∩ B)
    overlapping_genes = sorted(list(graph_nodes & proj_a_genes))
    
    print("\n==================================================")
    print(f"=== DISCOVERED OVERLAPPING GENES: {len(overlapping_genes)} ===")
    print("==================================================")
    
    if not overlapping_genes:
        print("Result: 0 genes directly overlapped between Project A concordant genes and the Project B literature graph.")
        print("Interpretation: Single-gene patient fold changes operate at a distinct resolution from high-level cellular aging abstracts.")
        return

    convergence_records = []
    for gene in overlapping_genes:
        deg = G.degree(gene, weight="weight")
        neighbors = list(G.neighbors(gene))
        
        autophagy_rel = G[gene]["autophagy"]["classification"] if G.has_edge(gene, "autophagy") else "No Direct Edge"
        senescence_rel = G[gene]["senescence"]["classification"] if G.has_edge(gene, "senescence") else "No Direct Edge"
        
        convergence_records.append({
            "Gene / Target": gene.upper(),
            "Graph Degree Weight": deg,
            "Connected Neighbors": len(neighbors),
            "Autophagy Relation": autophagy_rel,
            "Senescence Relation": senescence_rel
        })
        
    df_convergence = pd.DataFrame(convergence_records)
    print(df_convergence.to_string(index=False))
    
    # Save output
    output_path = "project-b-knowledge-graph/data/cross_project_convergent_targets.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_convergence.to_csv(output_path, index=False)
    print(f"\n[Storage] Saved non-circular alignment to: {output_path}")


if __name__ == "__main__":
    run_honest_cross_project_connection()