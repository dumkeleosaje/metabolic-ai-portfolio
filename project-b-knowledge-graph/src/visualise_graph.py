import os
import networkx as nx
from pyvis.network import Network
from build_graph import build_knowledge_graph


def generate_interactive_graph(min_paper_evidence=3, top_nodes=60):
    """
    Renders an interactive HTML knowledge graph matching the >=3 paper analysis threshold.
    """
    print("\n==================================================")
    print("=== GENERATING INTERACTIVE GRAPH VISUALISATION ===")
    print("==================================================")
    
    # 1. Build the base knowledge graph at threshold >= 3 papers
    G = build_knowledge_graph(min_paper_evidence=min_paper_evidence)
    
    # 2. Select top N most connected hubs
    degree_dict = dict(G.degree(weight="weight"))
    top_hub_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:top_nodes]
    G_sub = G.subgraph(top_hub_nodes).copy()
    
    print(f"Subsetting to Top {top_nodes} most connected biological hubs.")
    print(f"Visual graph contains: {G_sub.number_of_nodes()} nodes and {G_sub.number_of_edges()} edges.")
    
    net = Network(height="850px", width="100%", bgcolor="#0f172a", font_color="#f8fafc", cdn_resources="remote")
    
    for node in G_sub.nodes():
        weight = degree_dict.get(node, 10)
        node_size = max(12, min(50, int(weight / 25)))
        
        if node in ["autophagy", "senescence", "sasp", "oxidative stress"]:
            node_color = "#38bdf8"
        else:
            node_color = "#94a3b8"
            
        net.add_node(
            node,
            label=node.title(),
            title=f"Entity: {node.title()} (Paper Mentions: {weight})",
            size=node_size,
            color=node_color,
            font={"size": 15, "color": "#f8fafc"}
        )
        
    for u, v, data in G_sub.edges(data=True):
        classification = data.get("classification", "NEUTRAL")
        papers = data.get("pmid_count", 1)
        acts = data.get("activating_papers", 0)
        inhs = data.get("inhibiting_papers", 0)
        
        if classification == "CONTESTED":
            edge_color = "#f59e0b"
            title = f"CONTESTED: {u} <-> {v} ({acts} Act Papers / {inhs} Inh Papers, {papers} Total)"
        elif classification == "ACTIVATING":
            edge_color = "#22c55e"
            title = f"ACTIVATING: {u} -> {v} ({papers} Papers)"
        elif classification == "INHIBITING":
            edge_color = "#ef4444"
            title = f"INHIBITING: {u} -| {v} ({papers} Papers)"
        else:
            edge_color = "#475569"
            title = f"Co-occurrence / Nuanced: {papers} Papers"
            
        edge_width = max(1, min(6, int(papers / 3)))
        
        net.add_edge(
            u,
            v,
            color=edge_color,
            width=edge_width,
            title=title
        )
        
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.015,
          "springLength": 90,
          "springConstant": 0.08,
          "damping": 0.5
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 150 }
      },
      "nodes": {
        "borderWidth": 1,
        "borderWidthSelected": 3
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true
      }
    }
    """)
    
    output_dir = "project-b-knowledge-graph/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_html = os.path.join(output_dir, "knowledge_graph.html")
    
    net.write_html(output_html)
    print(f"\n[Visualisation Ready] Successfully generated: {output_html}")


if __name__ == "__main__":
    generate_interactive_graph(min_paper_evidence=3, top_nodes=60)