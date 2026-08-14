import os
import networkx as nx
from pyvis.network import Network
from build_graph import build_knowledge_graph


def generate_interactive_graph(min_paper_evidence=2, top_nodes=60):
    """
    Renders an interactive HTML knowledge graph with physics layout.
    Edges are color-coded:
      - Green: Consensus Activating
      - Red: Consensus Inhibiting
      - Amber/Orange: CONTESTED (Literature Disagreement)
    """
    print("\n==================================================")
    print("=== GENERATING INTERACTIVE GRAPH VISUALISATION ===")
    print("==================================================")
    
    # 1. Build the base knowledge graph
    G = build_knowledge_graph(min_paper_evidence=min_paper_evidence)
    
    # 2. Select top N most connected hubs to avoid visual clutter
    degree_dict = dict(G.degree(weight="weight"))
    top_hub_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:top_nodes]
    G_sub = G.subgraph(top_hub_nodes).copy()
    
    print(f"Subsetting to Top {top_nodes} most connected biological hubs.")
    print(f"Visual graph contains: {G_sub.number_of_nodes()} nodes and {G_sub.number_of_edges()} edges.")
    
    # 3. Initialize clean PyVis Network (no conflicting filter menus)
    net = Network(height="850px", width="100%", bgcolor="#0f172a", font_color="#f8fafc", cdn_resources="remote")
    
    # 4. Add Nodes with Custom Visual Styling
    for node in G_sub.nodes():
        weight = degree_dict.get(node, 10)
        node_size = max(12, min(50, int(weight / 25)))
        
        # Color key hubs distinctly
        if node in ["autophagy", "senescence", "aging", "sasp"]:
            node_color = "#38bdf8"  # Cyan for core hubs
        else:
            node_color = "#94a3b8"  # Slate grey for supporting entities
            
        net.add_node(
            node,
            label=node.title(),
            title=f"Entity: {node.title()} (Paper Mentions: {weight})",
            size=node_size,
            color=node_color,
            font={"size": 15, "color": "#f8fafc"}
        )
        
    # 5. Add Edges with Classification Colors
    for u, v, data in G_sub.edges(data=True):
        classification = data.get("classification", "NEUTRAL")
        papers = data.get("pmid_count", 1)
        acts = data.get("activating_signals", 0)
        inhs = data.get("inhibiting_signals", 0)
        
        if classification == "CONTESTED":
            edge_color = "#f59e0b"  # Amber / Orange for literature disputes
            title = f"CONTESTED: {u} <-> {v} ({acts} Act / {inhs} Inh, {papers} Papers)"
        elif classification == "ACTIVATING":
            edge_color = "#22c55e"  # Green
            title = f"ACTIVATING: {u} -> {v} ({papers} Papers)"
        elif classification == "INHIBITING":
            edge_color = "#ef4444"  # Red
            title = f"INHIBITING: {u} -| {v} ({papers} Papers)"
        else:
            edge_color = "#475569"  # Slate for neutral
            title = f"Co-occurrence: {papers} Papers"
            
        edge_width = max(1, min(6, int(papers / 3)))
        
        net.add_edge(
            u,
            v,
            color=edge_color,
            width=edge_width,
            title=title
        )
        
    # 6. Configure physics options directly
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
    
    # 7. Save to HTML
    output_dir = "project-b-knowledge-graph/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_html = os.path.join(output_dir, "knowledge_graph.html")
    
    net.write_html(output_html)
    print(f"\n[Visualisation Ready] Successfully generated: {output_html}")


if __name__ == "__main__":
    generate_interactive_graph(min_paper_evidence=2, top_nodes=60)