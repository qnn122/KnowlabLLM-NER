'''
Usage:

python eval.py \
    --filepath datasets/BC5CDR-disease_test.txt \
    --entity_type 'disease' \
    --output_dir 'results/phi3-4k-snomedct'

python eval.py \
    --filepath datasets/NCBI-disease_test.txt \
    --entity_type 'disease' \
    --output_dir 'results/phi3-4k-snomedct'

python eval.py \
    --filepath datasets/NLMGene_test.txt \
    --entity_type 'gene'

python eval.py \
    --filepath datasets/NLMChem_test.txt \
    --entity_type 'chemical'

python eval.py \
    --filepath datasets/BC2GM.txt \
    --entity_type 'gene'
'''

from tqdm import tqdm
from setup import get_answer, load_tokenizer, get_bio_tagging, bio_to_entities
#from nerval import crm
import pandas as pd
import os
import srsly
import fire
import eval4ner.muc as muc

def main(
    filepath: str,
    entity_type: str,
    output_dir: str
):
    # read lines from 'datasets/NCBI-disease_test.txt'
    print('>>> Processing file:', filepath)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Load tokenizer
    tokenizer = load_tokenizer()

    entity_type_short = entity_type[:3].upper()

    # Get true labels
    y_true = []
    for line in lines:
        tokens = tokenizer.tokenize(line)
        bio_tags = get_bio_tagging(tokens, entity_type_short=entity_type_short)
        y_true.append(bio_tags)


    # Get predictions
    lines_processed = [l.replace('<mark>', '').replace('</mark>', '') for l in lines]
    lines_pred = []
    for line in tqdm(lines_processed):
        try:
            ans = get_answer(line, entity_type='disease')
        except SyntaxError:
            ans = 'ERROR: SyntaxError'
        lines_pred.append(ans.strip())

    y_pred = []
    for line in lines_pred:
        tokens = tokenizer.tokenize(line)
        bio_tags = get_bio_tagging(tokens, entity_type_short=entity_type_short)
        y_pred.append(bio_tags)

    # Compute metrics
    y_true_ent = [y for y in map(bio_to_entities, [tokenizer.tokenize(line) for line in lines_processed], y_true)]
    y_pred_ent = [y for y in map(bio_to_entities, [tokenizer.tokenize(line) for line in lines_processed], y_pred)]
    results = muc.evaluate_all(y_pred_ent, y_true_ent * 1, lines_processed, verbose=False)
    df_result = pd.DataFrame(results).T
    df_result.iloc[:,:3] = round(df_result.iloc[:,:3]*100,3)
    df_result['count'] = df_result['count'].astype(int)
    print('>>> MUC Scores: ')
    print(df_result)

    # Save results
    path, filename = os.path.split(filepath)
    dataset_name, ext = os.path.splitext(filename)
    output_filepath_muc = os.path.join(output_dir, dataset_name + '_muc.csv')
    output_filepath_y_preds = os.path.join(output_dir, dataset_name + '_y_preds.jsonl')

    print('>>> Saving results to:', output_filepath_muc)
    df_result.to_csv(output_filepath_muc)
    print('>>> Saving predictions to:', output_filepath_y_preds)
    srsly.write_jsonl(output_filepath_y_preds, y_pred)


if __name__ == '__main__':
    fire.Fire(main)