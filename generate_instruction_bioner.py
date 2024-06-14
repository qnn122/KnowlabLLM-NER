import os
import random
import json

dataset_path = '../datasets'

datasets = {
    'BC2GM_train.txt': 'gene',
    'BC5CDR-chemical_train.txt': 'chemical',
    'NCBI-disease_train.txt': 'disease',
}

# read data from all datasets
dataset = []
for dataset_file in datasets:
    with open(os.path.join(dataset_path, dataset_file), 'r') as f:
        lines = f.readlines()
        # each data point is a tuple of (sentence, entity type)
        dataset.extend([(line, datasets[dataset_file]) for line in lines])

# shuffle the dataset
random.seed(42)
random.shuffle(dataset)

# generate instructions
output = []
for d in dataset:
    output_text = d[0]
    input_text = output_text.replace('<mark>', '').replace('</mark>', '')
    entity_type = d[1]
    data = {
        "instruction": f"Given a sentence, extract \"{entity_type}\" entities from it by highlighting them with <mark> and </mark>. If not present, output the same sentence. No further explanation needed.",
        "input": input_text,
        "output": output_text
    }
    output.append(data)

# save instructions to a file
with open('instructions_bioner.json', 'w') as f:
    json.dump(output, f, indent=4)


#