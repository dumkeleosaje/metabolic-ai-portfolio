import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from build_graph import canonicalize_entity, classify_scoped_relationship, build_knowledge_graph


def test_canonicalize_entity():
    """Verify entity cleaning and synonym resolution."""
    assert canonicalize_entity("cellular senescence") == "senescence"
    assert canonicalize_entity("autophagic flux") == "autophagy"
    assert canonicalize_entity("cell senescence") == "senescence"
    assert canonicalize_entity("senescence-associated secretory phenotype") == "sasp"
    assert canonicalize_entity("a") is None
    assert canonicalize_entity("") is None


def test_classify_scoped_relationship():
    """Verify between-span verb scoping and negation logic with precise character spans."""
    # 1. Clear Activating Case
    sent_act = "Rapamycin treatment promotes autophagy in human fibroblasts."
    span_a = [sent_act.find("Rapamycin"), sent_act.find("Rapamycin") + len("Rapamycin")]
    span_b = [sent_act.find("autophagy"), sent_act.find("autophagy") + len("autophagy")]
    assert classify_scoped_relationship(sent_act, span_a, span_b) == "ACTIVATING"
    
    # 2. Clear Inhibiting Case
    sent_inh = "GATA4 expression suppresses senescence in cells."
    span_a = [sent_inh.find("GATA4"), sent_inh.find("GATA4") + len("GATA4")]
    span_b = [sent_inh.find("senescence"), sent_inh.find("senescence") + len("senescence")]
    assert classify_scoped_relationship(sent_inh, span_a, span_b) == "INHIBITING"
    
    # 3. Negation Case ("does not activate" -> NEUTRAL / not classified as activation)
    sent_neg = "Compound X does not activate autophagy in this model."
    span_a = [sent_neg.find("Compound X"), sent_neg.find("Compound X") + len("Compound X")]
    span_b = [sent_neg.find("autophagy"), sent_neg.find("autophagy") + len("autophagy")]
    assert classify_scoped_relationship(sent_neg, span_a, span_b) == "NEUTRAL"
    
    # 4. Neutral / Co-occurrence without governing verb
    sent_neu = "We observed both mTOR and p53 expression in the nucleus."
    span_a = [sent_neu.find("mTOR"), sent_neu.find("mTOR") + len("mTOR")]
    span_b = [sent_neu.find("p53"), sent_neu.find("p53") + len("p53")]
    assert classify_scoped_relationship(sent_neu, span_a, span_b) == "NEUTRAL"


def test_real_knowledge_graph_structure():
    """Builds a test graph from actual extracted data and verifies graph properties."""
    G = build_knowledge_graph(min_paper_evidence=3)
    
    assert G.number_of_nodes() > 50
    assert G.number_of_edges() > 50
    assert G.has_node("autophagy")
    assert G.has_node("senescence")
    assert G.has_edge("autophagy", "senescence")
    
    edge_data = G["autophagy"]["senescence"]
    assert edge_data["classification"] in ["CONTESTED", "ACTIVATING", "INHIBITING", "NEUTRAL"]
    assert edge_data["pmid_count"] >= 3