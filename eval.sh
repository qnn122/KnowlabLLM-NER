python eval.py \
    --filepath datasets/NLMGene_test.txt \
    --entity_type 'gene' \
    --output_dir 'results/phi3-4k-snomedct'

python eval.py \
    --filepath datasets/NLMChem_test.txt \
    --entity_type 'chemical' \
    --output_dir 'results/phi3-4k-snomedct'

python eval.py \
    --filepath datasets/BC2GM.txt \
    --entity_type 'gene' \
    --output_dir 'results/phi3-4k-snomedct'