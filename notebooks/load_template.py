import yaml
from jinja2 import Template

# Function to load the YAML file
def load_prompt_from_yaml(file_path):
    with open(file_path, 'r') as file:
        prompt_dict = yaml.safe_load(file)
    return prompt_dict

# Load the prompt template from the YAML file
filepath = 'instruction_templates/NER.yaml'
prompt_dict = load_prompt_from_yaml(filepath)


prompt_template = prompt_dict['prompt']

# Example values for entity_type and input_text
example = {
    "entity_type": "person",
    "input_text": "Alice went to the store to meet Bob."
}


# Example values for entity_type and input_text
entity_type = "person"
input_text = "Alice went to the store to meet Bob."

# Create a Jinja2 template from the prompt
template = Template(prompt_template)

# Render the template with the given values
#rendered_prompt = template.render(entity_type=entity_type, input_text=input_text)

# Render the template with the given values
rendered_prompt = template.render(example)

print("Rendered Prompt:")
print(rendered_prompt)




# ===========================
examples = [
    {
        'input_text': 'glaucoma is one of the leading cause of blindness',
        'response': '<mark>glaucoma</mark> is one of the leading cause of blindness'
    },
    {
        'input_text': 'he has cataract',
        'response': 'he has <mark>cataract</mark>'
    }
]

new_input = 'inherited retinal disease can be more than glaucoma.'

prompt = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given a sentence, extract "disease" entities from it by highlighting them with <mark> and </mark>. If not present, output the same sentence.
### Input:
glaucoma is one of the leading cause of blindness
### Response:
<mark>glaucoma</mark> is one of the leading cause of blindness

### Input:
he has cataract
### Response:
he has <mark>cataract</mark>

### Input:
inherited retinal disease can be more than glaucoma.
### Response:

'''
# ======================================

# Get few-shot prompt template
prompt_template = prompt_dict['few_shot_prompt']
template = Template(prompt_template)
entity_type = 'disease'
rendered_prompt = template.render(entity_type=entity_type, examples=examples, new_input=new_input)
print(rendered_prompt)