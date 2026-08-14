import os
import json
from Bio import Entrez



def search_pubmed_ids(query, max_results=300, email="dumkele.osaje@gmail.cpm", api_key=None):
    """
    Searches PubMed for a keyword query and retrieves a list of unique PubMed IDs (PMIDs).
    
    Parameters:
    -----------
    query : str
        The search string (e.g., "autophagy AND senescence").
    max_results : int
        Maximum number of PMIDs to return (default: 300).
    email : str
        Your email address (required by NCBI to prevent blocking).
    api_key : str, optional
        Your NCBI API key to allow faster request limits.
        
    Returns:
    --------
    id_list : list of str
        List of PMIDs matching the search criteria.
    """
    # Step 1: Set identity credentials required by NCBI policy
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
        
    print(f"[PubMed Search] Querying PubMed for: '{query}' (Max results: {max_results})...")
    
    # Step 2: Query the PubMed search endpoint (esearch)
    # db="pubmed" specifies the literature database
    # term=query defines our search keywords
    # retmax specifies the number of IDs to retrieve
    # sort="relevance" returns the most relevant literature first
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance"
    )
    
    # Step 3: Parse the XML response into a Python dictionary
    record = Entrez.read(handle)
    handle.close()  # Always close the network stream handle
    
    # Step 4: Extract the list of PMIDs
    id_list = record["IdList"]
    
    print(f"[PubMed Search] Successfully retrieved {len(id_list)} PMIDs.")
    return id_list


def fetch_abstract_details(id_list, batch_size=100):
    """
    Takes a list of PubMed IDs (PMIDs), fetches their full XML records from NCBI,
    and extracts Title, Abstract, Year, and Journal into clean structured dictionaries.
    
    Parameters:
    -----------
    id_list : list of str
        List of PMIDs to retrieve.
    batch_size : int
        Number of records to fetch per network request (default: 100).
        
    Returns:
    --------
    articles_data : list of dict
        List of dictionaries, each containing:
        {'pmid': str, 'title': str, 'abstract': str, 'year': str, 'journal': str}
    """
    articles_data = []
    total_ids = len(id_list)
    
    print(f"[PubMed Fetch] Fetching detailed records for {total_ids} PMIDs...")
    
    # Process in batches to respect NCBI request size guidelines
    for i in range(0, total_ids, batch_size):
        batch_ids = id_list[i : i + batch_size]
        print(f"  Fetching batch {i + 1} to {min(i + batch_size, total_ids)} of {total_ids}...")
        
        # Step 1: Query NCBI efetch endpoint for full XML records
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch_ids),
            rettype="medline",
            retmode="xml"
        )
        
        # Step 2: Parse the XML response
        records = Entrez.read(handle)
        handle.close()
        
        # Step 3: Extract structured fields from each article record
        for article in records.get("PubmedArticle", []):
            medline = article.get("MedlineCitation", {})
            article_info = medline.get("Article", {})
            
            # Extract PMID
            pmid = str(medline.get("PMID", ""))
            
            # Extract Title
            title = article_info.get("ArticleTitle", "").strip()
            
            # Extract Abstract (handle multi-part structured abstracts)
            abstract_obj = article_info.get("Abstract", {})
            abstract_texts = abstract_obj.get("AbstractText", [])
            
            if isinstance(abstract_texts, list):
                # Join sections (e.g., Background, Methods, Results) into one text string
                abstract = " ".join([str(txt) for txt in abstract_texts]).strip()
            else:
                abstract = str(abstract_texts).strip()
                
            # Defensive check: skip articles that do not have an abstract text
            if not abstract:
                continue
                
            # Extract Publication Year defensively
            journal_issue = article_info.get("Journal", {}).get("JournalIssue", {})
            pub_date = journal_issue.get("PubDate", {})
            year = pub_date.get("Year", pub_date.get("MedlineDate", "Unknown"))
            
            # Extract Journal Name
            journal = article_info.get("Journal", {}).get("Title", "Unknown Journal")
            
            articles_data.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": str(year)[:4],  # Keep 4-digit year format
                "journal": journal
            })
            
    print(f"[PubMed Fetch] Successfully parsed {len(articles_data)} articles containing valid abstracts.")
    return articles_data

def save_abstracts_to_json(articles_data, output_filepath):
    """
    Saves the list of parsed article dictionaries to a structured JSON file.
    Creates parent directories automatically if they do not exist.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(articles_data, f, indent=2, ensure_ascii=False)
        
    print(f"[Storage] Saved {len(articles_data)} records to: {output_filepath}")

def main():
    # 1. Define query and output destination
    query = "autophagy AND senescence"
    output_path = "project-b-knowledge-graph/data/pubmed_autophagy_senescence.json"
    
    # 2. Fetch PMIDs
    pmids = search_pubmed_ids(query=query, max_results=300, email="student@bath.ac.uk")
    
    # 3. Download abstract texts
    articles = fetch_abstract_details(pmids, batch_size=100)
    
    # 4. Save to local disk
    save_abstracts_to_json(articles, output_path)
    
    # 5. Sanity Check: Print sample 1st abstract to verify content
    if articles:
        print("\n--- SAMPLE FETCHED RECORD [1/300] ---")
        print(f"PMID:    {articles[0]['pmid']}")
        print(f"Title:   {articles[0]['title']}")
        print(f"Year:    {articles[0]['year']}")
        print(f"Abstract Preview: {articles[0]['abstract'][:200]}...")


if __name__ == "__main__":
    main()