import yaml
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from colorama import init, Fore, Style

def load_yaml_template(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def render_template(template_str, messages, add_generation_prompt=True):
    template = Template(template_str)
    return template.render(messages=messages, add_generation_prompt=add_generation_prompt)

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

def chat_with_model(model, tokenizer, template_str):
    print("Start chatting with the model! Type 'exit' to stop.")

    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        conversation_history.append({"role": "user", "content": user_input})

        prompt = render_template(template_str, conversation_history)

        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        params = {        
            "max_new_tokens": 200,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.15,
        }

        outputs = model.generate(**inputs, **params, return_dict_in_generate=True, output_scores=False)
        generated_sequences = outputs.sequences

        # Decode the generated tokens to get the output text
        generated_text = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)[0]

        # Extract the assistant's response (after the last "Assistant: " occurrence)
        assistant_response = generated_text.split("Assistant:")[-1].strip()

        print(f"Assistant: {assistant_response}")

        conversation_history.append({"role": "assistant", "content": assistant_response})

def chat_with_model(model, tokenizer, template_str):
    print("Start chatting with the model! Type 'exit' to stop.")

    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        conversation_history.append({"role": "user", "content": user_input})
        print_ui("user", user_input)

        prompt = render_template(template_str, conversation_history)

        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        params = {        
            "max_new_tokens": 200,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.15,
        }

        outputs = model.generate(**inputs, **params, return_dict_in_generate=True, output_scores=False)
        generated_sequences = outputs.sequences

        # Decode the generated tokens to get the output text
        generated_text = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)[0]

        # Extract the assistant's response, assuming it starts after the last user input
        #assistant_response = generated_text.split("[/INST]")[-1].strip()
        assistant_response = generated_text.split("[/INST]")[-1].strip()

        print_ui("assistant", assistant_response)

        conversation_history.append({"role": "assistant", "content": assistant_response})


def print_ui(role, text):
    separator = "=" * 50
    if role == "user":
        print(f"\n{separator}\nYou: {text}\n{separator}")
    elif role == "assistant":
        print(f"\n{separator}\nAssistant: {text}\n{separator}\n")

#============================================================================
def print_ui(role, text):
    separator = "=" * 50
    if role == "user":
        print(f"{separator}\n{Fore.GREEN}You: {text}\n{separator}")
    elif role == "assistant":
        print(f"{separator}\n{Fore.CYAN}Assistant: {text}\n{separator}\n")

def chat_with_model(model, tokenizer, template_str):
    print("Start chatting with the model! Type 'exit' to stop.")

    conversation_history = []

    while True:
        user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
        if user_input.lower() == 'exit':
            break

        conversation_history.append({"role": "user", "content": user_input})
        #print_ui("user", user_input)

        prompt = render_template(template_str, conversation_history)

        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        params = {        
            "max_new_tokens": 200,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.15,
        }

        outputs = model.generate(**inputs, **params, return_dict_in_generate=True, output_scores=False)
        generated_sequences = outputs.sequences

        # Decode the generated tokens to get the output text
        generated_text = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)[0]

        # Extract the assistant's response, assuming it starts after the last user input
        assistant_response = generated_text.split("[/INST]")[-1].strip()
        #assistant_response = generated_text

        print_ui("assistant", assistant_response)

        conversation_history.append({"role": "assistant", "content": assistant_response})


#================================================================================

if __name__ == "__main__":
    template_data = load_yaml_template("Llama-v2_2.yaml")
    template_str = template_data['instruction_template']

    checkpoint_path = "models/Meta-Llama-3-8B"
    checkpoint_path = "models/Phi-3-mini-4k-instruct"
    model, tokenizer = load_models_tokenizer(checkpoint_path)
    chat_with_model(model, tokenizer, template_str)
