import json
import os
import re
from collections import Counter
from itertools import combinations
import spacy
import scispacy

# Import the single source of truth for synonyms
from build_graph import SYNONYM_MAP

# Comprehensive blacklist to eliminate non-biological, methodological, and academic noise
COMPREHENSIVE_STOPWORDS = {
    # Academic & Paper Structure Terms
    "study", "studies", "result", "results", "patient", "patients", "role",
    "mechanism", "mechanisms", "method", "methods", "effect", "effects",
    "activity", "activities", "expression", "level", "levels", "data",
    "analysis", "group", "groups", "cell", "cells", "tissue", "tissues",
    "model", "models", "investigate", "progression", "process", "processes",
    "function", "functions", "author", "authors", "conclusion", "conclusions",
    "review", "potential", "findings", "target", "targets", "targeted",
    "therapeutic", "therapy", "treatment", "treatments", "novel", "evidence",
    "insights", "understanding", "associated", "associated with", "research",
    "involvement", "assessed", "detected", "investigated", "characterized",
    
    # Methodological, Experimental & Organismal Terms
    "human", "mice", "mouse", "rat", "rats", "in vitro", "in vivo", "ex vivo",
    "western blotting", "knockdown", "overexpression", "assay", "assays",
    "staining", "culture", "cultured", "sample", "samples", "cohort",
    
    # Non-specific Phenotypic, Temporal & Spatial Adjectives/Nouns
    "phenotype", "phenotypes", "damaged", "damage", "changes", "markers",
    "marker", "age", "aging", "aged", "cellular", "intracellular", "extracellular",
    "development", "response", "responses", "pathway", "pathways", "factor", "factors",
    "metabolic", "pathogenesis", "degradation", "production", "secretion", "secretory",
    "decline",
    
    # Verbs, Actions & Relational States
    "increased", "decreased", "inhibition", "activation", "induced", "induce",
    "inducing", "activated", "inhibited", "activating", "inhibiting", "suppressed",
    "suppression", "impaired", "impairment", "enhanced", "enhancement", "attenuated",
    "attenuation", "reduced", "reduction", "accumulation", "accumulated", "mediating",
    "mediated", "regulation", "regulated", "regulating", "increase", "decrease", "block", "blocks",
    "blocked", "blocking", "prevents", "prevented", "preventing", "promoting", "promotes",
    "promote", "targeting", "induction", "upregulation"
}

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def strip_html_tags(text):
    """Removes HTML/XML tags and normalises whitespace."""
    if not text:
        return ""
    clean_text = HTML_TAG_PATTERN.sub("", text)
    return " ".join(clean_text.split())


def clean_entity(raw_text):
    """Cleans, lowercases, filters noise, and canonicalizes synonyms."""
    if not raw_text:
        return None
    text = raw_text.strip().lower()
    
    if len(text) <= 2:
        return None
    if text.replace(".", "").replace("-", "").isdigit():
        return None
    if text in COMPREHENSIVE_STOPWORDS:
        return None
        
    return SYNONYM_MAP.get(text, text)


def extract_cooccurrences_from_doc(doc, pmid, year):
    """Extracts entity pairs with exact character spans."""
    extracted_records = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        
        sent_entities = []
        for ent in sent.ents:
            cleaned = clean_entity(ent.text)
            if cleaned:
                sent_entities.append({
                    "text": cleaned,
                    "start_char": ent.start_char - sent.start_char,
                    "end_char": ent.end_char - sent.start_char
                })
                
        unique_entities = {}
        for item in sent_entities:
            if item["text"] not in unique_entities:
                unique_entities[item["text"]] = item
                
        entity_list = list(unique_entities.values())
        
        if len(entity_list) >= 2:
            for ent_a, ent_b in combinations(entity_list, 2):
                extracted_records.append({
                    "entity_a": ent_a["text"],
                    "entity_b": ent_b["text"],
                    "span_a": [ent_a["start_char"], ent_a["end_char"]],
                    "span_b": [ent_b["start_char"], ent_b["end_char"]],
                    "evidence_sentence": sent_text,
                    "pmid": pmid,
                    "year": year
                })
                
    return extracted_records


def run_entity_extraction_pipeline():
    print("\n==================================================")
    print("=== SCISPACY ENTITY & CO-OCCURRENCE EXTRACTION ===")
    print("=== (UNIFIED VOCABULARY & STRIPPED HTML) ===")
    print("==================================================")
    
    input_json = "project-b-knowledge-graph/data/pubmed_autophagy_senescence.json"
    output_json = "project-b-knowledge-graph/data/extracted_entity_pairs.json"
    
    with open(input_json, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    print(f"Loaded {len(articles)} abstracts from disk.")
    nlp = spacy.load("en_core_sci_sm")
    
    all_pairs = []
    entity_frequency = Counter()
    
    for i, article in enumerate(articles, 1):
        pmid = article.get("pmid", "")
        year = article.get("year", "")
        raw_abstract = article.get("abstract", "")
        
        if not raw_abstract:
            continue
            
        clean_abstract = strip_html_tags(raw_abstract)
        doc = nlp(clean_abstract)
        
        for ent in doc.ents:
            cleaned = clean_entity(ent.text)
            if cleaned:
                entity_frequency[cleaned] += 1
                
        pairs = extract_cooccurrences_from_doc(doc, pmid, year)
        all_pairs.extend(pairs)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
        
    print(f"[Storage] Saved {len(all_pairs)} clean pairs to: {output_json}")
    
    print("\n==================================================")
    print("=== TOP 20 CANONICAL BIOLOGICAL ENTITIES ===")
    print("==================================================")
    for rank, (entity, count) in enumerate(entity_frequency.most_common(20), 1):
        print(f"  {rank:2d}. {entity:<30} (Count: {count})")


if __name__ == "__main__":
    run_entity_extraction_pipeline()