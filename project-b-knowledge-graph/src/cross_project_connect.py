import os
import pandas as pd
import networkx as nx
from build_graph import build_knowledge_graph


def run_cross_project_connection():
    print("\n==================================================")
    print("=== CROSS-PROJECT MOLECULAR CONVERGENCE ===")
    print("=== (PROJECT A TRANSCRIPTOMICS vs PROJECT B GRAPH) ===")
    print("==================================================")
    
    # 1. Build Project B Knowledge Graph
    G = build_knowledge_graph(min_paper_evidence=2)
    graph_nodes = set(node.lower() for node in G.nodes())
    print(f"Project B Knowledge Graph contains {len(graph_nodes)} unique entities.")
    
    # 2. Load Project A Muscle-Liver Concordant Genes
    proj_a_genes_path = "project-a-convergence-classifier/data/muscle_liver_concordant_genes.csv"
    
    if os.path.exists(proj_a_genes_path):
        df_proj_a = pd.read_csv(proj_a_genes_path)
        proj_a_genes = set(df_proj_a["Gene_Symbol"].astype(str).str.lower().tolist())
        print(f"Loaded {len(proj_a_genes)} concordant genes from Project A.")
    else:
        print(f"[Warning] {proj_a_genes_path} not found. Using curated core metabolic genes.")
        proj_a_genes = {"mtor", "irs1", "akt1", "pparg", "ampk", "sirt1", "foxo1", "nr4a1", "junb", "fosl1"}

    # 3. Add canonical metabolic/insulin signaling genes from Project A literature
    canonical_metabolic_genes = {"mtor", "ampk", "sirt1", "akt", "akt1", "irs1", "pparg", "foxo1", "p53", "beclin-1"}
    combined_query_genes = proj_a_genes.union(canonical_metabolic_genes)
    
    # 4. Find Overlapping Genes
    overlapping_genes = sorted(list(graph_nodes & combined_query_genes))
    print(f"\nDiscovered {len(overlapping_genes)} Convergent Genes present in both Project A & Project B!")
    
    # 5. Inspect Graph Neighborhood for Convergent Genes
    convergence_records = []
    
    for gene in overlapping_genes:
        deg = G.degree(gene, weight="weight")
        neighbors = list(G.neighbors(gene))
        
        # Check interactions with key hubs
        autophagy_rel = G[gene]["autophagy"]["classification"] if G.has_edge(gene, "autophagy") else "No Direct Edge"
        senescence_rel = G[gene]["senescence"]["classification"] if G.has_edge(gene, "senescence") else "No Direct Edge"
        
        convergence_records.append({
            "Gene / Target": gene.upper(),
            "Total Paper Weight": deg,
            "Connected Neighbors": len(neighbors),
            "Autophagy Relation": autophagy_rel,
            "Senescence Relation": senescence_rel
        })
        
    df_convergence = pd.DataFrame(convergence_records)
    print("\n==================================================")
    print("=== CONVERGENT MOLECULAR TARGETS TABLE ===")
    print("==================================================")
    print(df_convergence.to_string(index=False))
    
    # 6. Save Cross-Project Alignment Table
    output_path = "project-b-knowledge-graph/data/cross_project_convergent_targets.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_convergence.to_csv(output_path, index=False)
    print(f"\n[Storage] Saved convergence analysis to: {output_path}")


if __name__ == "__main__":
    run_cross_project_connection()