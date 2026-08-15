import json
import os
from collections import Counter
from itertools import combinations
import spacy
import scispacy

# Comprehensive list to eliminate non-biological, methodological, and academic noise
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
    "insights", "understanding", "associated", "associated with",
    
    # Methodological, Experimental & Organismal Terms
    "human", "mice", "mouse", "rat", "rats", "in vitro", "in vivo", "ex vivo",
    "western blotting", "knockdown", "overexpression", "assay", "assays",
    "staining", "culture", "cultured", "sample", "samples", "cohort",
    
    # Non-specific Phenotypic, Temporal & Spatial Adjectives
    "phenotype", "phenotypes", "damaged", "damage", "changes", "markers",
    "marker", "age", "aging", "cellular", "intracellular", "extracellular",
    "development", "response", "responses", "pathway", "pathways", "factor", "factors",
    
    # Verbs, Actions & Relational States (Relational tokens are captured as edges, not nodes)
    "increased", "decreased", "inhibition", "activation", "induced", "induce",
    "inducing", "activated", "inhibited", "activating", "inhibiting", "suppressed",
    "suppression", "impaired", "impairment", "enhanced", "enhancement", "attenuated",
    "attenuation", "reduced", "reduction", "accumulation", "accumulated", "mediating",
    "mediated", "regulation", "regulated", "increase", "decrease", "block", "blocks",
    "blocked", "blocking", "prevents", "prevented", "preventing", "promoting", "promotes"
}


def clean_entity(raw_text):
    """
    Cleans raw entity strings, normalizes whitespace, and filters out noise.
    """
    text = raw_text.strip().lower()
    
    # Filter 1: Drop empty strings or short abbreviations (<= 2 characters)
    if len(text) <= 2:
        return None
        
    # Filter 2: Drop pure numbers or punctuation-heavy tokens
    if text.replace(".", "").replace("-", "").isdigit():
        return None
        
    # Filter 3: Drop blacklisted academic, methodological, or relational noise
    if text in COMPREHENSIVE_STOPWORDS:
        return None
        
    return text


def extract_cooccurrences_from_doc(doc, pmid, year):
    """
    Splits an abstract into sentences and extracts entity pairs co-occurring
    in the same sentence alongside character offsets and the evidence sentence.
    """
    extracted_records = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        
        # Extract valid entities within this sentence while capturing character spans
        sent_entities = []
        for ent in sent.ents:
            cleaned = clean_entity(ent.text)
            if cleaned:
                sent_entities.append({
                    "text": cleaned,
                    "start_char": ent.start_char - sent.start_char,
                    "end_char": ent.end_char - sent.start_char
                })
                
        # Deduplicate entities in the same sentence by name (keep first occurrence span)
        unique_entities = {}
        for item in sent_entities:
            if item["text"] not in unique_entities:
                unique_entities[item["text"]] = item
                
        entity_list = list(unique_entities.values())
        
        # Extract pairs if 2 or more distinct valid entities co-occur
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
    print("==================================================")
    
    input_json = "project-b-knowledge-graph/data/pubmed_autophagy_senescence.json"
    output_json = "project-b-knowledge-graph/data/extracted_entity_pairs.json"
    
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input file not found: {input_json}. Run pubmed_fetch.py first.")
        
    with open(input_json, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    print(f"Loaded {len(articles)} abstracts from disk.")
    print("Loading SciSpacy biomedical language model ('en_core_sci_sm')...")
    nlp = spacy.load("en_core_sci_sm")
    
    all_pairs = []
    entity_frequency = Counter()
    
    print("\nProcessing abstracts and filtering vocabulary...")
    for i, article in enumerate(articles, 1):
        pmid = article.get("pmid", "")
        year = article.get("year", "")
        abstract_text = article.get("abstract", "")
        
        if not abstract_text:
            continue
            
        doc = nlp(abstract_text)
        
        for ent in doc.ents:
            cleaned = clean_entity(ent.text)
            if cleaned:
                entity_frequency[cleaned] += 1
                
        pairs = extract_cooccurrences_from_doc(doc, pmid, year)
        all_pairs.extend(pairs)
        
        if i % 50 == 0 or i == len(articles):
            print(f"  [Progress] Processed {i}/{len(articles)} abstracts ({len(all_pairs)} filtered pairs extracted)...")

    # Save filtered pairs
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
        
    print(f"\n[Storage] Saved {len(all_pairs)} filtered pairs to: {output_json}")
    
    print("\n==================================================")
    print("=== TOP 20 FILTERED BIOLOGICAL ENTITIES ===")
    print("==================================================")
    for rank, (entity, count) in enumerate(entity_frequency.most_common(20), 1):
        print(f"  {rank:2d}. {entity:<30} (Count: {count})")


if __name__ == "__main__":
    run_entity_extraction_pipeline()