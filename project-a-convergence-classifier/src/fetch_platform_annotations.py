import os
import gzip
import urllib.request
import urllib.error
import pandas as pd

def fetch_gpl_annotation(gpl_id, output_path):
    """Download and extract probe-to-gene mapping table for a GEO Platform (GPL)."""
    print(f"Fetching platform annotation for {gpl_id}...")
    
    # URL candidates across GEO FTP structure
    urls = [
        f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{gpl_id[:-3]}nnn/{gpl_id}/annot/{gpl_id}.annot.gz",
        f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{gpl_id[:-3]}nnn/{gpl_id}/soft/{gpl_id}.soft.gz",
        f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl_id}&targ=self&form=text&view=data"
    ]
    
    success = False
    
    for url in urls:
        try:
            if url.endswith(".gz"):
                gz_path = f"{output_path}.gz"
                urllib.request.urlretrieve(url, gz_path)
                with gzip.open(gz_path, 'rt', encoding='utf-8', errors='ignore') as f_in:
                    with open(output_path, 'w', encoding='utf-8') as f_out:
                        for line in f_in:
                            if not line.startswith("!"):
                                f_out.write(line)
                if os.path.exists(gz_path):
                    os.remove(gz_path)
            else:
                # Direct text stream fallback
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    content = response.read().decode('utf-8', errors='ignore')
                    lines = [l for l in content.splitlines() if not l.startswith("!")]
                    with open(output_path, 'w', encoding='utf-8') as f_out:
                        f_out.write("\n".join(lines))
            
            # Check if file was created and contains valid content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"  [SUCCESS] Downloaded and saved {gpl_id} table to: {output_path}")
                success = True
                break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            print(f"  [INFO] Endpoint {url} failed: {e}. Trying fallback...")
            continue
            
    if not success:
        raise RuntimeError(f"Failed to fetch annotation table for {gpl_id} across all GEO endpoints.")

if __name__ == "__main__":
    data_dir = "project-a-convergence-classifier/data"
    fetch_gpl_annotation("GPL570", f"{data_dir}/GPL570.annot")     # GSE10946 (Affymetrix)
    fetch_gpl_annotation("GPL14951", f"{data_dir}/GPL14951.annot") # GSE89632 (Illumina)