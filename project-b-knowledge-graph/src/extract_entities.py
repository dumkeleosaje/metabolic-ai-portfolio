import json
import os
from collections import Counter
from itertools import combinations
import spacy
import scispacy

# Academic stopwords to filter out non-specific biological terms
GENERIC_STOPWORDS = {
    "study", "studies", "result", "results", "patient", "patients", "role",
    "mechanism", "mechanisms", "method", "methods", "effect", "effects",
    "activity", "activities", "expression", "level", "levels", "data",
    "analysis", "group", "groups", "cell", "cells", "tissue", "tissues",
    "model", "models", "investigate", "progression", "process", "processes",
    "function", "functions", "author", "authors", "conclusion", "conclusions"
}


def clean_entity(raw_text):
    """
    Cleans raw entity strings and filters out generic noise.
    """
    text = raw_text.strip().lower()
    
    # Drop short noise tokens or pure numbers
    if len(text) <= 2 or text.replace(".", "").isdigit():
        return None
        
    # Drop non-specific academic words
    if text in GENERIC_STOPWORDS:
        return None
        
    return text


def extract_cooccurrences_from_doc(doc, pmid, year):
    """
    Splits an abstract document into sentences and extracts all entity pairs
    co-occurring within the same sentence alongside the evidence text.
    """
    extracted_records = []
    
    # Iterate sentence by sentence through the abstract
    for sent in doc.sents:
        sent_text = sent.text.strip()
        
        # Extract and clean all entities present within this specific sentence
        valid_entities = []
        for ent in sent.ents:
            cleaned = clean_entity(ent.text)
            if cleaned and cleaned not in valid_entities:
                valid_entities.append(cleaned)
                
        # If 2 or more distinct biological entities appear in the same sentence,
        # create co-occurrence pair records
        if len(valid_entities) >= 2:
            for ent_a, ent_b in combinations(sorted(valid_entities), 2):
                extracted_records.append({
                    "entity_a": ent_a,
                    "entity_b": ent_b,
                    "evidence_sentence": sent_text,
                    "pmid": pmid,
                    "year": year
                })
                
    return extracted_records


def run_entity_extraction_pipeline():
    print("\n==================================================")
    print("=== SCISPACY ENTITY & CO-OCCURRENCE EXTRACTION ===")
    print("==================================================")
    
    # 1. Load the 295 fetched PubMed abstracts
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
    
    print("\nProcessing abstracts and segmenting sentences...")
    for i, article in enumerate(articles, 1):
        pmid = article.get("pmid", "")
        year = article.get("year", "")
        abstract_text = article.get("abstract", "")
        
        if not abstract_text:
            continue
            
        doc = nlp(abstract_text)
        
        # Track global entity frequency
        for ent in doc.ents:
            cleaned = clean_entity(ent.text)
            if cleaned:
                entity_frequency[cleaned] += 1
                
        # Extract sentence-level pairs
        pairs = extract_cooccurrences_from_doc(doc, pmid, year)
        all_pairs.extend(pairs)
        
        if i % 50 == 0 or i == len(articles):
            print(f"  [Progress] Processed {i}/{len(articles)} abstracts ({len(all_pairs)} pairs extracted)...")

    # 2. Save extracted pairs to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
        
    print(f"\n[Storage] Successfully saved {len(all_pairs)} interaction pairs to: {output_json}")
    
    # 3. Print Top 20 Most Mentioned Biological Entities
    print("\n==================================================")
    print("=== TOP 20 EXTRACTED BIOLOGICAL ENTITIES ===")
    print("==================================================")
    for rank, (entity, count) in enumerate(entity_frequency.most_common(20), 1):
        print(f"  {rank:2d}. {entity:<30} (Count: {count})")


if __name__ == "__main__":
    run_entity_extraction_pipeline()