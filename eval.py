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
from setup import get_answer, load_tokenizer, get_bio_tagging, bio_to_entities, promptify
#from nerval import crm
import pandas as pd
import os
import srsly
import fire
import eval4ner.muc as muc
import random
import yaml
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
#from setup import promptify

def load_models_tokenizer(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True
    )
    return model, tokenizer

def get_answer_checkpoint(input_text, entity_type, template, examples, model, tokenizer):    
    prompt = promptify(input_text, entity_type, template, examples)

    # if prompt does not end with \n\n, add it
    if not prompt.endswith('\n'):
        prompt += '\n'

    inputs = tokenizer([prompt], return_tensors="pt")
    params = {        
        "max_new_tokens": 400,
        #"temperature": 0.1,
        #"do_sample": True,
        #"repetition_penalty": 1.15,
        #"guidance_scale": 1
    }
    #outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
    outputs = model.generate(**inputs, **params)

    outputs_gen = outputs[0][inputs['input_ids'].shape[1]:]
    output_text = tokenizer.decode(outputs_gen)
    #output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


    return output_text


# Function to load the YAML file
def load_prompt_from_yaml(file_path='templates/NER.yaml'):
    with open(file_path, 'r') as file:
        template = yaml.safe_load(file)
    return template


def main(
    filepath: str,
    entity_type: str,
    output_dir: str,
    n_examples: int = None,
    seed: int = 42,
    n_samples: int = None,
    checkpoint_path: str = None
):
    # read lines from 'datasets/NCBI-disease_test.txt'
    print('>>> Processing file:', filepath)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if checkpoint_path:
        checkpoint_path = "models/Phi-3-mini-4k-instruct"
        model, tokenizer = load_models_tokenizer(checkpoint_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # Load tokenizer
        tokenizer = load_tokenizer()

    # add special token <mark> and </mark> to the tokenizer if not already present
    if '<mark>' not in tokenizer.special_tokens_map.values():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<mark>', '</mark>']})


    entity_type_short = entity_type[:3].upper()

    # Prompt template
    template = load_prompt_from_yaml()

    # Sample for quicker evaluation
    if n_samples:
        random.seed(seed)
        lines = random.sample(lines, n_samples)

    # Get true labels
    y_true = []
    for line in lines:
        tokens = tokenizer.tokenize(line)
        bio_tags = get_bio_tagging(tokens, entity_type_short=entity_type_short)
        y_true.append(bio_tags)

    # Get example for few-shot learning
    if n_examples:
        filepath_train = filepath.replace('test', 'train')
        with open(filepath_train, 'r') as f:
            lines_train = f.readlines()

        random.seed(seed)
        examples_tmp = random.sample(lines_train, n_examples)
        examples = [{'input_text': e.replace('<mark>', '').replace('</mark>', ''), 'response': e} for e in examples_tmp]
    else:
        examples = None

    
    # Get predictions
    lines_processed = [l.replace('<mark>', '').replace('</mark>', '') for l in lines]
    lines_pred = []
    for i, line in enumerate(tqdm(lines_processed)):
        #line = line.strip()
        try:
            if checkpoint_path:
                ans = get_answer_checkpoint(line, entity_type, template, examples, model, tokenizer)
            else:
                ans = get_answer(line, entity_type, template, examples)
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
    output_filepath_lines_pred = os.path.join(output_dir, dataset_name + '_lines_preds.jsonl')
    output_filepath_lines_data = os.path.join(output_dir, dataset_name + '_lines_data.jsonl')
    ouput_prompt_template = os.path.join(output_dir, dataset_name + '_prompt_template.txt')

    # make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # saving
    print('>>> Saving results to:', output_filepath_muc)
    df_result.to_csv(output_filepath_muc)
    print('>>> Saving predictions to:', output_filepath_y_preds)
    srsly.write_jsonl(output_filepath_y_preds, y_pred)
    srsly.write_jsonl(output_filepath_lines_pred, lines_pred)
    srsly.write_jsonl(output_filepath_lines_data, lines)

    prompt = promptify(line, entity_type, template, examples)
    config = {
        'entity_type': entity_type,
        'n_samples': n_samples,
        'prompt': prompt
    }
    # write config as json with indent=4
    with open(ouput_prompt_template, 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)