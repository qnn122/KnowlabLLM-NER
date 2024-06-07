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

checkpoint_path = "models/Phi-3-mini-4k-instruct"
model, tokenizer = load_models_tokenizer(checkpoint_path)


input_text = "Who is Barack Obama?"
tokenizer.pad_token_id = tokenizer.eos_token_id
inputs = tokenizer([input_text], return_tensors="pt")


params = {        
    "max_new_tokens": 200,
    "temperature": 0.7,
    "do_sample": True,
    "repetition_penalty": 1.15,
    #"guidance_scale": 1
}
#outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
outputs = model.generate(**inputs, **params)


output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
output_text = output_text[len(input_text) :]
print(f"Input: {input_text}\nCompletion: {output_text}")


def chat_with_model(model, tokenizer):
    print("\nWelcome to the chat interface! Type 'exit' to stop the chat.\n")
    
    while True:
        # Get input from the user
        input_text = input("You: ")
        
        if input_text.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            break
        
        # Tokenize the input
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer([input_text], return_tensors="pt")

        # Define generation parameters
        params = {        
            "max_new_tokens": 200,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.15,
        }

        # Generate response from the model
        outputs = model.generate(**inputs, **params, return_dict_in_generate=True, output_scores=False)
        generated_sequences = outputs.sequences

        # Decode the generated tokens to text
        generated_text = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)[0]
        
        # Print the model's response
        print(f"Model: {generated_text}\n")


def chat_with_model_template(model, tokenizer):
    print("Start chatting with the model! Type 'exit' to stop.")

    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        conversation_history.append(f"User: {user_input}")

        # Join conversation history into a single string with separator tokens
        prompt = "\n".join(conversation_history) + "\nAssistant:"

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

        conversation_history.append(f"Assistant: {assistant_response}")


#chat_with_model(model, tokenizer)
chat_with_model_template(model, tokenizer)