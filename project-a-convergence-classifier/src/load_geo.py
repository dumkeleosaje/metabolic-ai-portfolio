import pandas as pd

def parse_metadata(file_path):

    #CREATE an empty list for sample IDs
    sample_ids = []
    #CREATE an empty list for disease labels
    disease_labels = []

    #open the file for reading using a with block for automatic closing
    with open(file_path, "r") as file:
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
            #DISCARD the first element (it's the marker itself)
            #STRIP surrounding quotation marks from each remaining element
            #STORE these as the sample IDs
            for item in raw_ids:
                clean_ids = item.strip().strip('"').strip()
                sample_ids.append(clean_ids)

        #IF the line starts with the sample-characteristics marker:
        if line.startswith("!Sample_characteristics_ch1") and "glycemiagroup:" in line:
            parts = line.split("\t") #split by tabs
            raw_labels = parts[1:]
            #SPLIT the line on tabs
            #DISCARD the first element
            
            for item in raw_labels:
                clean_item = item.strip().strip('"').strip()
                
                if ":" in clean_item:
                    label_value = clean_item.split(":")[-1].strip()
                else:
                    label_value = clean_item

                disease_labels.append(label_value)
                #STRIP quotes
                #STRIP the prefix before the colon, keeping only the value
                #STORE these as the disease labels
            #IF no:
                #SKIP — this characteristics line is age or sex, not disease

        #IF the line marks the start of the matrix:
            #STOP reading — no more metadata below this point

    #CHECK that sample IDs and disease labels have the same length
    if len(sample_ids) != len(disease_labels):
    #IF they don't:
        #RAISE an error with a clear message
        raise ValueError("Error: Number of sample IDs does not match number of disease labels")

    patient_map = {}
    for i in range(len(sample_ids)):
        raw = disease_labels[i]
        if raw == "1":
            label = "NGT"
        elif raw == "2":
            label = "IGT"
        elif raw == "3":
            label = "DM"
        else:
            label = raw
            
        patient_map[sample_ids[i]] = label
        
    return patient_map

    #BUILD a mapping from each sample ID to its disease label
    #RETURN that mapping


def parse_matrix(file_path):

#USE pandas to read the file as tab-separated text
        #TELL it to treat lines starting with "!" as comments
        #TELL it to use the first column as the row index
    df = pd.read_csv(file_path, sep="\t", comment="!", index_col=0)

    #remove surrounding quotes and spaces from column names 
    df.columns = [col.strip().strip('"').strip() for col in df.columns]

    #return the DataFrame
    return df
    

    #CHECK the result:
        #Is the last row junk (from the table-end marker)?
        #IF so, remove it

    #CHECK that all values are numeric
        #IF any column is text, something has gone wrong — investigate

    #PRINT the shape so you can see what you got

def load_dataset(file_path):
    #Main loader function that parses both metadata and expression matrix

    #Get metadata mapping dictionary
    metadata_map = parse_metadata(file_path)

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
    X, y = load_dataset(test_file_path)
    
    print("\n[SUCCESS] Loaded Dataset Summary:")
    print(f"Expression Matrix (X) Shape: {X.shape}")
    print(f"Labels Series (y) Shape:     {y.shape}")
    print("\nClass Distribution:")
    print(y.value_counts())