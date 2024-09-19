# ===============================
OUPUT_DIR='results/mixtral-7x8B-4bit-5shot-samples'

python eval.py \
    --filepath datasets/BC2GM_test.txt \
    --entity_type 'gene' \
    --output_dir $OUPUT_DIR \
    --n_examples 5 \
    --n_samples 200

python eval.py \
    --filepath datasets/BC5CDR-chemical_test.txt \
    --entity_type 'chemical' \
    --output_dir $OUPUT_DIR \
    --n_examples 5 \
    --n_samples 200

python eval.py \
    --filepath datasets/NCBI-disease_test.txt \
    --entity_type 'disease' \
    --output_dir $OUPUT_DIR \
    --n_examples 5 \
    --n_samples 200



# ===============================
OUPUT_DIR='results/phi3-4k-base-5shot-samples-finetuned'

python eval.py \
    --filepath datasets/BC2GM_test.txt \
    --entity_type 'gene' \
    --output_dir $OUPUT_DIR \
    --n_examples 5 \
    --n_samples 200 \
    --checkpoint_path stanford_alpaca/models/Phi-3-mini-4k-instruct_finetuned/checkpoint-150

python eval.py \
    --filepath datasets/BC5CDR-chemical_test.txt \
    --entity_type 'chemical' \
    --output_dir $OUPUT_DIR \
    --n_examples 5 \
    --n_samples 200 \
    --checkpoint_path stanford_alpaca/models/Phi-3-mini-4k-instruct_finetuned/checkpoint-150

python eval.py \
    --filepath datasets/NCBI-disease_test.txt \
    --entity_type 'disease' \
    --output_dir $OUPUT_DIR \
    --n_examples 5 \
    --n_samples 200 \
    --checkpoint_path stanford_alpaca/models/Phi-3-mini-4k-instruct_finetuned/checkpoint-150


# ===============================
OUPUT_DIR='results/phi3-4k-wikiterms-5shot-samples-finetuned-api'

python eval.py \
    --filepath datasets/BC2GM_test.txt \
    --entity_type 'gene' \
    --output_dir $OUPUT_DIR \
    --n_examples 5 \
    --n_samples 200 

python eval.py \
    --filepath datasets/BC5CDR-chemical_test.txt \
    --entity_type 'chemical' \
    --output_dir $OUPUT_DIR \
    --n_examples 5 \
    --n_samples 200 

python eval.py \
    --filepath datasets/NCBI-disease_test.txt \
    --entity_type 'disease' \
    --output_dir $OUPUT_DIR \
    --n_examples 5 \
    --n_samples 200 

# ===============================
OUPUT_DIR='results/phi3-4k-wikiterms-0shot-samples-finetuned-api'

python eval.py \
    --filepath datasets/BC2GM_test.txt \
    --entity_type 'gene' \
    --output_dir $OUPUT_DIR \
    --n_samples 200 

python eval.py \
    --filepath datasets/BC5CDR-chemical_test.txt \
    --entity_type 'chemical' \
    --output_dir $OUPUT_DIR \
    --n_samples 200 

python eval.py \
    --filepath datasets/NCBI-disease_test.txt \
    --entity_type 'disease' \
    --output_dir $OUPUT_DIR \
    --n_samples 200 


### 
OUPUT_DIR='results/debug'

python eval.py \
    --filepath datasets/BC2GM_test.txt \
    --entity_type 'gene' \
    --output_dir $OUPUT_DIR \
    --n_samples 200 



# ===============================
MODEL="KnowMedPhi3-medium-it-finetuned"

MODEL="KnowMedPhi3-mini-it-finetuned"
OUPUT_DIR="results/${MODEL}-0shot-samples-finetuned"
CHECKPOINT="/home/quangng/LLM/KnowlabLLM-NER/stanford_alpaca/models/${MODEL}/checkpoint-50"
python eval.py \
    --filepath datasets/BC2GM_test.txt \
    --entity_type 'gene' \
    --output_dir $OUPUT_DIR \
    --n_samples 200 \
    --checkpoint_path $CHECKPOINT

python eval.py \
    --filepath datasets/BC5CDR-chemical_test.txt \
    --entity_type 'chemical' \
    --output_dir $OUPUT_DIR \
    --n_samples 200 \
    --checkpoint_path $CHECKPOINT

python eval.py \
    --filepath datasets/NCBI-disease_test.txt \
    --entity_type 'disease' \
    --output_dir $OUPUT_DIR \
    --n_samples 200 \
    --checkpoint_path $CHECKPOINT


# ALL
MODEL="KnowMedPhi3-mini-it-finetuned"
OUPUT_DIR="results/all/${MODEL}-0shot-samples-finetuned"
CHECKPOINT="/home/quangng/LLM/KnowlabLLM-NER/stanford_alpaca/models/${MODEL}/checkpoint-100"
python eval.py \
    --filepath datasets/BC2GM_test.txt \
    --entity_type 'gene' \
    --output_dir $OUPUT_DIR \
    --checkpoint_path $CHECKPOINT

python eval.py \
    --filepath datasets/BC5CDR-chemical_test.txt \
    --entity_type 'chemical' \
    --output_dir $OUPUT_DIR \
    --checkpoint_path $CHECKPOINT

python eval.py \
    --filepath datasets/NCBI-disease_test.txt \
    --entity_type 'disease' \
    --output_dir $OUPUT_DIR \
    --checkpoint_path $CHECKPOINT