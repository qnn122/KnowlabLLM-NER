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
from modules.utils import get_ents, get_answer, load_tokenizer, get_bio_tagging, bio_to_entities, promptify
#from nerval import crm
import pandas as pd
import os
import srsly
import fire
import eval4ner.muc as muc
import random
import yaml
import json
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
#from setup import promptify
import torch.nn.functional as F

from modules import sampler_hijack
from modules.callbacks import _StopEverythingStoppingCriteria

from transformers import LogitsProcessorList, LogitsProcessor

sampler_hijack.hijack_samplers()

# def decode(output_ids, skip_special_tokens=True):
#     if shared.tokenizer is None:
#         raise ValueError('No tokenizer is loaded')

#     return shared.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)


# def get_reply_from_output_ids(output_ids, state=None, starting_from=0):
#     reply = decode(output_ids[starting_from:], state['skip_special_tokens'] if state else True)

#     # Handle tokenizers that do not add the leading space for the first token
#     if (hasattr(shared.tokenizer, 'convert_ids_to_tokens') and len(output_ids) > starting_from) and not reply.startswith(' '):
#         first_token = shared.tokenizer.convert_ids_to_tokens(int(output_ids[starting_from]))
#         if isinstance(first_token, (bytes,)):
#             first_token = first_token.decode('utf8')

#         if first_token.startswith('▁'):
#             reply = ' ' + reply

#     return reply

def decode(output_ids, tokenizer, skip_special_tokens=True):
    if tokenizer is None:
        raise ValueError('No tokenizer is loaded')

    return tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)


def get_reply_from_output_ids(output_ids, tokenizer, state=None, starting_from=0):
    reply = decode(output_ids[starting_from:], state['skip_special_tokens'] if state else True)

    # Handle tokenizers that do not add the leading space for the first token
    if (hasattr(tokenizer, 'convert_ids_to_tokens') and len(output_ids) > starting_from) and not reply.startswith(' '):
        first_token = tokenizer.convert_ids_to_tokens(int(output_ids[starting_from]))
        if isinstance(first_token, (bytes,)):
            first_token = first_token.decode('utf8')

        if first_token.startswith('▁'):
            reply = ' ' + reply

    return reply

class LogprobProcessor(LogitsProcessor):
    def __init__(self, logprobs=None):
        self.logprobs = logprobs
        self.token_alternatives = {}

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        if self.logprobs is not None:  # 0-5
            log_e_probabilities = F.log_softmax(logits, dim=1)
            top_values, top_indices = torch.topk(log_e_probabilities, k=self.logprobs + 1)
            top_tokens = [get_reply_from_output_ids([tok]) for tok in top_indices[0]]
            top_probs = [float(x) for x in top_values[0]]
            self.token_alternatives = dict(zip(top_tokens, top_probs))

        return logits

    def __repr__(self):
        return f"<{self.__class__.__name__}(logprobs={self.logprobs}, token_alternatives={self.token_alternatives})>"



def load_models_tokenizer(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = 32000
    tokenizer.eos_token_id = 32007
    # model = AutoModelForCausalLM.from_pretrained(
    #     checkpoint_path,
    #     pad_token_id=tokenizer.pad_token_id,
    #     device_map="auto",
    #     trust_remote_code=True,
    #     low_cpu_mem_usage=True, 
    #     torch_dtype=torch.float16, 
    # ).eval()
    params_model = {
        'low_cpu_mem_usage': True, 
        'torch_dtype': torch.float16, 
        'trust_remote_code': True
    }
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        **params_model
    )
    # model.generation_config = GenerationConfig.from_pretrained(
    #     checkpoint_path,
    #     pad_token_id=tokenizer.pad_token_id,
    #     trust_remote_code=True
    # )
    model = model.cuda()
    return model, tokenizer

def get_answer_checkpoint(input_text, entity_type, template, examples, model, tokenizer):    
    prompt = promptify(input_text, entity_type, template, examples)

    # if prompt does not end with \n\n, add it
    if not prompt.endswith('\n'):
        prompt += '\n'

    inputs = tokenizer([prompt], return_tensors="pt")
    # params = {        
    #     "max_new_tokens": 400,
    #     #"temperature": 0.1,
    #     #"do_sample": True,
    #     #"repetition_penalty": 1.15,
    #     #"guidance_scale": 1
    # }

    generate_params = {
        'max_new_tokens': 200, 
        'temperature': 0.1, 'temperature_last': False, 
        'dynamic_temperature': False, 'dynatemp_low': 1, 'dynatemp_high': 1, 'dynatemp_exponent': 1, 
        'smoothing_factor': 0, 'smoothing_curve': 1, 
        'top_p': 1, 'min_p': 0, 'top_k': 0, 
        'repetition_penalty': 1.15, 'presence_penalty': 0, 
        'frequency_penalty': 0, 'repetition_penalty_range': 1024, 
        'typical_p': 1.0, 'tfs': 1, 'top_a': 0, 'guidance_scale': 1.0, 
        'penalty_alpha': 0, 'mirostat_mode': 0, 'mirostat_tau': 5, 'mirostat_eta': 0.1, 
        'do_sample': True, 'encoder_repetition_penalty': 1, 'no_repeat_ngram_size': 0, 
        'use_cache': True,
        'eos_token_id': [32007] # tokenizer.eos_token_id
    }

    generate_params['stopping_criteria'] = transformers.StoppingCriteriaList()
    generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria())

    logprobs = None  # coming to chat eventually
    processor = LogprobProcessor(logprobs)
    processor = LogitsProcessorList([processor])
    generate_params['logits_processor'] = processor

    #outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
    # make sure inputs in cuda too
    #inputs = {k: v.cuda() for k, v in inputs.items()}
    
    generate_params['inputs'] = inputs['input_ids'].cuda()

    output = model.generate(**generate_params)[0]

    output_gen = output[inputs['input_ids'].shape[1]:]
    output_text = tokenizer.decode(output_gen)
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
        model, tokenizer = load_models_tokenizer(checkpoint_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # Load tokenizer
        tokenizer = load_tokenizer()

    # add special token <mark> and </mark> to the tokenizer if not already present
    if '<mark>' not in tokenizer.special_tokens_map.values():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<mark>', '</mark>']})


    # Prompt template
    template = load_prompt_from_yaml()

    # Sample for quicker evaluation
    if n_samples:
        random.seed(seed)
        lines = random.sample(lines, n_samples)

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

    # post process
    entity_type_short = entity_type[:3].upper()
    '''
    y_true = []
    for line in lines:
        tokens = tokenizer.tokenize(line)
        bio_tags = get_bio_tagging(tokens, entity_type_short=entity_type_short)
        y_true.append(bio_tags)
    y_pred = []
    for line in lines_pred:
        tokens = tokenizer.tokenize(line)
        bio_tags = get_bio_tagging(tokens, entity_type_short=entity_type_short)
        y_pred.append(bio_tags)

    y_true_ent = [y for y in map(bio_to_entities, [tokenizer.tokenize(line) for line in lines_processed], y_true)]
    y_pred_ent = [y for y in map(bio_to_entities, [tokenizer.tokenize(line) for line in lines_processed], y_pred)]
    '''
    y_true_ent = [[('CHEM', entity) for entity in get_ents(line)] for line in lines]
    y_pred_ent = [[('CHEM', entity) for entity in get_ents(line)] for line in lines_pred]

    # Compute metrics
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
    #output_filepath_y_preds = os.path.join(output_dir, dataset_name + '_y_preds.jsonl')
    output_filepath_lines_pred = os.path.join(output_dir, dataset_name + '_lines_preds.jsonl')
    output_filepath_lines_data = os.path.join(output_dir, dataset_name + '_lines_data.jsonl')
    ouput_prompt_template = os.path.join(output_dir, dataset_name + '_prompt_template.txt')

    # make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # saving
    print('>>> Saving results to:', output_filepath_muc)
    df_result.to_csv(output_filepath_muc)
    #print('>>> Saving predictions to:', output_filepath_y_preds)
    #srsly.write_jsonl(output_filepath_y_preds, y_pred)
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