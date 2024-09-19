#!/bin/bash

# Define the base directory for datasets
data_dir="datasets"

# Define an associative array mapping filenames to entity types
declare -A entity_types
entity_types=(
    ["BC2GM"]="gene"
    ["BC5CDR-chemical"]="chemical"
    ["BC5CDR-disease"]="disease"
    ["NCBI-disease"]="disease"
    ["NLMChem"]="chemical"
    ["NLMGene"]="gene"
)

# Define an array of filenames (without extensions)
filenames=(cp
    "BC2GM_test" 
    "BC5CDR-chemical_test" 
    #"BC5CDR-disease_test" 
    "NCBI-disease_test" 
    #"NLMChem_test" 
    #"NLMGene_test"
)

# Output directory
DATA_NAME="pathbank" # CHANGE HERE
OUPUT_DIR="results/phi3-4k-${DATA_NAME}-0shot-samples-finetuned"
CHECKPOINT="/home/quangng/LLM/KnowlabLLM-NER/stanford_alpaca/models/phi-3-mini-4k-${DATA_NAME}-instruct-finetuned/checkpoint-100"

# Loop over the filenames and run the command
for filename in "${filenames[@]}"; do
    # Extract the base name (without the "_test" part) to use as the key for entity_types
    base_name="${filename%_test}"
    
    python eval.py \
        --filepath "$data_dir/$filename.txt" \
        --entity_type "${entity_types[$base_name]}" \
        --output_dir "$OUPUT_DIR" \
        --n_samples 200 \
        --checkpoint_path $CHECKPOINT
done
