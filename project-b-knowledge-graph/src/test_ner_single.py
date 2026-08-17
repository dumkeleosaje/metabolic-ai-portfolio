import json
import spacy
import scispacy


def test_single_abstract_ner():
    print("\n=== TESTING SCISPACY NER ON SINGLE ABSTRACT ===")
    
    # 1. Load the first abstract from JSON
    json_path = "project-b-knowledge-graph/data/pubmed_autophagy_senescence.json"
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    first_article = articles[0]
    print(f"Title: {first_article['title']}")
    print(f"PMID:  {first_article['pmid']}\n")
    
    # 2. Load the SciSpacy biomedical language model
    print("Loading 'en_core_sci_sm' model...")
    nlp = spacy.load("en_core_sci_sm")
    
    # 3. Process the abstract text
    doc = nlp(first_article["abstract"])
    
    # 4. Print extracted biological entities
    print(f"\n--- Extracted Entities ({len(doc.ents)} found) ---")
    for i, ent in enumerate(doc.ents[:15], 1):  # Show top 15
        print(f"  {i:2d}. {ent.text:<30} (Label: {ent.label_})")


if __name__ == "__main__":
    test_single_abstract_ner()