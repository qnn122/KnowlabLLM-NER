# ===============================
python eval.py \
    --filepath datasets/BC2GM_test.txt \
    --entity_type 'gene' \
    --output_dir 'results/phi3-4k-base' \
    --n_examples 5

python eval.py \
    --filepath datasets/BC5CDR-chemical_test.txt \
    --entity_type 'chemical' \
    --output_dir 'results/phi3-4k-base' \
    --n_examples 5

python eval.py \
    --filepath datasets/NCBI-disease_test.txt \
    --entity_type 'gene' \
    --output_dir 'results/phi3-4k-base' \
    --n_examples 5
