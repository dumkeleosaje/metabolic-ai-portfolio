import pandas as pd
import requests
import io


def fetch_biomart_table(url, server_name):
    """Query BioMart endpoint for transcript-to-HGNC mapping."""
    print(f"Querying {server_name} BioMart API...")

    xml_query = """<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE Query>
    <Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "1" count = "" datasetConfigVersion = "0.6">
        <Dataset name = "hsapiens_gene_ensembl" interface = "default" >
            <Attribute name = "ensembl_transcript_id" />
            <Attribute name = "hgnc_symbol" />
        </Dataset>
    </Query>
    """

    # Use GET request with query parameter to comply with Ensembl API rules
    response = requests.get(url, params={'query': xml_query})

    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text), sep="\t")
        
        # Robustness check: Catch silent BioMart HTML error responses returning 200 OK
        if df.shape[0] < 10000 or "Transcript stable ID" not in df.columns:
            raise ValueError(f"CRITICAL ERROR: {server_name} returned an incomplete or invalid BioMart response table!")
            
        df.columns = ["Transcript stable ID", "HGNC symbol"]
        df = df.dropna(subset=["HGNC symbol"]).copy()
        df["HGNC symbol"] = df["HGNC symbol"].str.strip()
        df = df[df["HGNC symbol"] != ""]
        print(f"  [SUCCESS] Retrieved {len(df)} mapped entries from {server_name}.")
        return df
    else:
        raise RuntimeError(f"Failed to query {server_name}. Status code: {response.status_code}")


def build_combined_biomart_map(output_path="project-a-convergence-classifier/data/enst_to_hgnc_map.tsv"):
    """Fetch modern (GRCh38) and legacy (GRCh37) BioMart mappings and combine them."""
    url_modern = "https://www.ensembl.org/biomart/martservice"
    url_legacy = "https://grch37.ensembl.org/biomart/martservice"

    df_modern = fetch_biomart_table(url_modern, "Modern Ensembl (GRCh38)")
    df_legacy = fetch_biomart_table(url_legacy, "Legacy Ensembl (GRCh37)")

    print("\nMerging modern and legacy BioMart dictionaries...")
    # Modern takes priority; legacy acts as fallback for retired IDs
    combined_df = pd.concat([df_modern, df_legacy], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["Transcript stable ID"], keep="first")

    modern_unique = len(df_modern.drop_duplicates(subset=["Transcript stable ID"]))
    legacy_added = len(combined_df) - modern_unique
    print(f"  Modern-only unique mapped transcripts: {modern_unique}")
    print(f"  Legacy GRCh37 fallback transcripts added: {legacy_added}")

    combined_df.to_csv(output_path, sep="\t", index=False)
    print(f"\n=== COMBINED MAP CREATED ===")
    print(f"Total Unique Mapped Transcripts: {len(combined_df)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    build_combined_biomart_map()