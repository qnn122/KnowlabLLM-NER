import yaml
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from colorama import init, Fore, Style

import streamlit as st
from streamlit_chat import message
import os
import time

def load_yaml_template(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

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


def generate_output_single(model, tokenizer, template_str, user_input):
    conversation_history = [{"role": "user", "content": user_input}]
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
    return assistant_response, generated_text

def generate_ouput_conversation(model, tokenizer, template_str, user_input, conversation_history):
    #conversation_history.append({"role": "user", "content": user_input})
    prompt = render_template(template_str, conversation_history)
    #prompt = "[INST] <<SYS>>\nAnswer the questions.\n<</SYS>>\n\nwho are you? [/INST]"
    #prompt = f"[INST] <<SYS>>\n{template_str}\n<</SYS>>\n\n{user_input} [/INST]"


    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # params = {        
    #     "max_new_tokens": 200,
    #     "temperature": 0.7,
    #     "do_sample": True,
    #     "repetition_penalty": 1.15,
    # }

    params = {        
        "max_new_tokens": 500
    }
    outputs = model.generate(**inputs, **params, return_dict_in_generate=True, output_scores=False)
    generated_sequences = outputs.sequences

    # Decode the generated tokens to get the output text
    generated_text = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)[0]

    # Extract the assistant's response, assuming it starts after the last user input
    assistant_response = generated_text.split("[/INST]")[-1].strip()
    #assistant_response = generated_text

    #conversation_history.append({"role": "assistant", "content": assistant_response})
    return assistant_response, generated_text, prompt


#===========================================
#                APP
#=============================================== 
st.set_page_config(layout="wide")
# Initialize streamlit app and side bart to select model and template

# Initialize session state
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'generated_full' not in st.session_state:
    st.session_state['generated_full'] = []
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

# get subfolder in models
options = os.listdir("models")

# select model
with st.sidebar:
    # select the model, default: "models/Phi-3-mini-4k-instruct"
    model_name = st.selectbox("Model", options, index=len(options)-1)

    # select template file as a text input, default: `Llama-v2_2.yaml`
    template_file = st.text_input("Template file", "Llama-v2_2.yaml")

    # load button
    if st.button("Load model"):
        checkpoint_path = f"models/{model_name}"
        model, tokenizer = load_models_tokenizer(checkpoint_path)

        template_data = load_yaml_template(template_file)
        template_str = template_data['instruction_template']

        # store in session state
        st.session_state['model'] = model 
        st.session_state['tokenizer'] = tokenizer
        st.session_state['template_str'] = template_str
        st.session_state['model_loaded'] = True

        # Display reminder if model is not loaded
    if not st.session_state.get('model_loaded'):
        st.warning("Please load the model to start using the app.")

    # Display message if model is loaded
    if st.session_state.get('model_loaded'):
        st.write("Model and template loaded successfully")
        #st.write(st.session_state['template_str'])



#st.session_state.setdefault('past', ['hi'])
#st.session_state.setdefault('generated', ["Hello, I'm a language model. How can I help you?"])
#st.session_state.setdefault('generated_full', ["Hello, I'm a language model. How can I help you?"])

def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    
    assistant_response, generated_full, prompt = generate_ouput_conversation(
        st.session_state['model'],
        st.session_state['tokenizer'],
        st.session_state['template_str'],
        user_input,
        st.session_state['conversation_history']
    )
    
    st.session_state.generated.append(assistant_response)
    st.session_state.generated_full.append(generated_full)  # For full conversation history
    #st.session_state.conversation_history = updated_history
    print("\n=====================\n", prompt)

def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]
    del st.session_state.generated_full[:]
    st.session_state.conversation_history = []

# Streamed response emulator
def response_generator(response, delay=0.05):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Layout with columns
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Chat with Language Model")

    chat_placeholder = st.empty()

    with chat_placeholder.container():    
       #for i in range(min(len(st.session_state['past']), len(st.session_state['generated']))): 
            # User              
            # message(
            #     st.session_state['past'][i], 
            #     is_user=True, 
            #     key=f"{i}_user")
            # # Agent
            # message(
            #     st.session_state['generated'][i], 
            #     key=f"{i}", 
            #     allow_html=True
            # )
            # with st.chat_message('user'):
            #     st.markdown(st.session_state['past'][i])
            # with st.chat_message('assistant'):
            #     st.write_stream(response_generator(st.session_state['generated'][i]))
        for message in st.session_state['conversation_history']:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                #st.write_stream(response_generator(message['content']))
        

    #with st.container():
    #    if st.session_state.get('model_loaded'):
    #        st.chat_input("User Input:", on_submit=on_input_change, key="user_input")
        # Accept user input
        if st.session_state.get('model_loaded') and (prompt := st.chat_input("User Input:", on_submit=on_input_change, key="user_input")):
            # Add user message to chat history
            st.session_state['conversation_history'].append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = st.write_stream(response_generator(st.session_state.generated[-1]))
                #print(response)
            st.session_state['conversation_history'].append({"role": "assistant", "content": response})
        
        st.button("Clear message", on_click=on_btn_click)

with col2:
    st.header("Model Inspection")
    st.write(st.session_state['generated_full'])