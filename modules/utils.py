import requests
from ast import literal_eval
from transformers import AutoTokenizer
import yaml
from jinja2 import Template



def promptify(input_text, entity_type, template, examples=None):
    if examples:
        prompt_template = Template(template['few_shot'])
        PROMPT = prompt_template.render(entity_type=entity_type, examples=examples, input_text=input_text)
    else:
        prompt_template = Template(template['zero_shot'])
        PROMPT = prompt_template.render(entity_type=entity_type, input_text=input_text)
    return PROMPT


def load_tokenizer(model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # new tokens
    new_tokens = ["<mark>", "</mark>"]

    # check if the tokens are already in the vocabulary
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer

def get_response(text):
    url = "http://127.0.0.1:5000/v1/completions"
    headers = {
        "Content-Type": "application/json"
    }

    params = {
        "prompt": text,
        "max_tokens": 200,
        "temperature": 0.1,
        "typical_p": 1,
        "seed": 10,
        "return_full_text": False,
        "repetition_penalty": 1.15,
        "repetition_penalty_range":1024,
        "guidance_scale": 1,
        "skip_special_tokens": False
    }

    response = requests.post(url, headers=headers, json=params)

    answer = literal_eval(response.text)['choices'][0]['text']
    #answer = response.text
    return answer

def get_answer(input_text, entity_type, template, examples=None):
    prompt = promptify(input_text, entity_type, template, examples)

    # if prompt does not end with \n\n, add it
    if not prompt.endswith('\n'):
        prompt += '\n'
    #return literal_eval(answer)
    return get_response(prompt)


def get_ents(line):
    '''Just get entities between <mark> and </mark> in line
    assuming that there are more than one entity in a line
    '''
    entities = []
    start = 0
    while True:
        start = line.find('<mark>', start)
        if start == -1:
            break
        end = line.find('</mark>', start)
        entity = line[start + 6: end].strip()
        entities.append(entity)
        start = end
    return entities

# token next to <mark> becomes B-<entity_type_short>
# tokens from B-<entity_type_short> to one before </mark> becomes I-<entity_type_short>
# other tokens are O
# remore <mark> and </mark> from tokens 
def get_bio_tagging(tokens, entity_type_short='DIS'):
    tags = ['O'] * len(tokens)

    # mark <mark> and </mark> tokens as 'MS' and 'ME' in the bio_tagging
    tags = ['MS' if token == '<mark>' else 'ME' if token == '</mark>' else tag for (tag, token) in zip(tags, tokens)]

    i = 0
    n = len(tags)
    
    while i < n:
        if tags[i] == 'MS':
            # Start from the next token after 'MS'
            start = i + 1
            while start < n and tags[start] != 'ME':
                start += 1
            
            # Now 'start' should be at 'ME' or out of bounds
            if start < n and tags[start] == 'ME':
                if start - i == 2:
                    # Only one token between 'MS' and 'ME'
                    tags[i + 1] = 'B' + '-' + entity_type_short
                else:
                    # More than one token between 'MS' and 'ME'
                    tags[i + 1] = 'B' + '-' + entity_type_short
                    for j in range(i + 2, start):
                        tags[j] = 'I' + '-' + entity_type_short
            i = start # Continue from the end of this segment
        else:
            i += 1

    # remove all 'MS' and 'ME' tokens
    tags = [tag for tag in tags if tag not in ['MS', 'ME']]
    
    return tags


def bio_to_entities(tokens, tokens_bio):
    entities, entity, label = [], [], None
    for token, bio in zip(tokens, tokens_bio):
        if bio.startswith('B-'):
            if entity: entities.append((label, ' '.join(entity)))
            entity, label = [token], bio[2:]
        elif bio.startswith('I-') and label:
            entity.append(token)
        else:
            if entity: entities.append((label, ' '.join(entity)))
            entity, label = [], None
    if entity: entities.append((label, ' '.join(entity)))
    return entities


def print_muc_scores(results):
    print('\n', 'NER evaluation scores:')
    for mode, res in results.items():
        print("{:>8s} mode, Precision={:<6.4f}, Recall={:<6.4f}, F1:{:<6.4f}"
                .format(mode, res['precision'], res['recall'], res['f1_score']))