import pandas as pd
import numpy as np

def parse_metadata(file_path, target_keyword="glycemiagroup"):

    #CREATE an empty list for sample IDs
    sample_ids = []
    #CREATE an empty list for disease labels
    disease_labels = []

    #open the file for reading using a with block for automatic closing with utf-8 encoding guard
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    #FOR each line in the file:
    for line in lines:
        
        #stop if we hit the data table (Fixed bug: added missing '!')
        if line.startswith("!series_matrix_table_begin"):
            break 

        #IF the line does not start with "!":
        if not line.startswith("!"):
            continue
            #SKIP it (we only want metadata in this pass)

        #IF the line starts with the sample-accession marker:
        if line.startswith("!Sample_geo_accession"):    
            #SPLIT the line on tabs
            parts = line.split("\t")
            raw_ids = parts[1:]
            for item in raw_ids:
                clean_ids = item.strip().strip('"').strip()
                sample_ids.append(clean_ids)

        # Handle GSE10946 title-based parsing (PCOS vs nonPCOS in !Sample_title)
        if (target_keyword.lower() in ["title", "pcos"]) and line.startswith("!Sample_title"):
            parts = line.split("\t")[1:]
            for item in parts:
                clean_item = item.strip().strip('"').strip()
                if "nonpcos" in clean_item.lower():
                    disease_labels.append("Control")
                elif "pcos" in clean_item.lower():
                    disease_labels.append("PCOS")
                else:
                    disease_labels.append(clean_item)

        #IF the line starts with the sample-characteristics marker:
        elif line.startswith("!Sample_characteristics_ch1") and target_keyword.lower() in line.lower():
            parts = line.split("\t") #split by tabs
            raw_labels = parts[1:]
            
            for item in raw_labels:
                clean_item = item.strip().strip('"').strip()
                
                # Handle comma-separated single lines (e.g. GSE10946)
                if "," in clean_item and target_keyword.lower() in clean_item.lower():
                    sub_parts = clean_item.split(",")
                    found_val = clean_item
                    for sp in sub_parts:
                        if target_keyword.lower() in sp.lower():
                            found_val = sp.strip()
                            break
                    clean_item = found_val

                if ":" in clean_item:
                    label_value = clean_item.split(":")[-1].strip()
                else:
                    label_value = clean_item

                disease_labels.append(label_value)

    #CHECK that sample IDs and disease labels have the same length
    if len(sample_ids) != len(disease_labels):
        raise ValueError(f"Error: Number of sample IDs ({len(sample_ids)}) does not match number of disease labels ({len(disease_labels)}) for keyword '{target_keyword}'")

    # Guard against single-class constant label assignment
    unique_labels = set(disease_labels)
    if len(unique_labels) <= 1:
        raise ValueError(
            f"CRITICAL LABEL ERROR: Target keyword '{target_keyword}' produced a single unique value ({unique_labels}). "
            f"Classification targets must contain at least two classes!"
        )

    patient_map = {}
    for i in range(len(sample_ids)):
        raw = disease_labels[i]
        if target_keyword == "glycemiagroup":
            if raw == "1":
                label = "NGT"
            elif raw == "2":
                label = "IGT"
            elif raw == "3":
                label = "DM"
            else:
                label = raw
        else:
            label = raw
            
        patient_map[sample_ids[i]] = label
        
    return patient_map


def parse_matrix(file_path):
    df = pd.read_csv(file_path, sep="\t", comment="!", index_col=0, encoding="utf-8")
    df.columns = [col.strip().strip('"').strip() for col in df.columns]
    return df
    

def load_dataset(file_path, target_keyword="characteristics_ch1"):

    #Get metadata mapping dictionary
    metadata_map = parse_metadata(file_path, target_keyword=target_keyword)

    #Get expression DataFrame
    df_matrix = parse_matrix(file_path)

    #Convert metadata dictionary into a Pandas Series 
    labels_series = pd.Series(metadata_map, name="disease_group")

    # Explicit Validation Step: ensure sample counts and order match
    if len(labels_series) != df_matrix.shape[1]:
        raise ValueError(f"Sample count mismatch: Metadata has {len(labels_series)} samples, Matrix has {df_matrix.shape[1]} columns.")

    # Check for missing sample IDs in matrix columns
    missing_samples = set(labels_series.index) - set(df_matrix.columns)
    if missing_samples:
        raise KeyError(f"The following metadata sample IDs are missing from expression matrix: {missing_samples}")

    df_matrix = df_matrix[labels_series.index]

    return df_matrix, labels_series 


if __name__ == "__main__":
    test_file_path = "project-a-convergence-classifier/data/GSE18732_series_matrix.txt"
    
    print("Testing full pipeline via load_dataset()...")
    X, y = load_dataset(test_file_path, target_keyword="glycemiagroup")
    
    print("\n[SUCCESS] Loaded Dataset Summary:")
    print(f"Expression Matrix (X) Shape: {X.shape}")
    print(f"Labels Series (y) Shape:     {y.shape}")
    print("\nClass Distribution:")
    print(y.value_counts())