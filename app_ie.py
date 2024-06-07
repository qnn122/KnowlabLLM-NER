import streamlit as st
import random 

st.set_page_config(layout="wide")

# ========================================
# SIDEBAR
# ========================================
with st.sidebar:
    # model path
    model_name = st.text_input("Model Name", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")

    # model parameters
    max_tokens = st.number_input("Max Tokens", 500)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

    # input filepath
    filepath = st.text_input("Filepath", "datasets/NCBI-disease_test.txt")
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # RUN ALL button
    if st.button("RUN ALL"):
        st.write("Running all components...")
    
    # Save filepath
    filepath_save = st.text_input("Save Filepath", "output/NCBI-disease_test_pred.txt")



# ========================================
# MAIN PAGE
# ========================================
row1_col1, row1_col2 = st.columns(2)

# TOP-LEFT quarter: components of prompt
with row1_col1:
    st.header("Prompt Components")

    # instruction
    instruction = st.text_area("Instruction", "Given a sentence, extract 'disease' entities from it by highlighting them with <mark> and </mark>. If not present, output the same sentence.")

    # output format
    output_format = st.text_area("Output Format", "<mark>glaucoma</mark> is one of the leading cause of blindness")

    # examples
    examples = st.text_area("Examples", "glaucoma is one of the leading cause of blindness\nhe has cataract")

# Add content to the second quarter (second column of the first row)
with row1_col2:
    st.header("Quarter 2")
    st.button("Click me")
    st.write("This is a button.")

# Create the second row with two columns
row2_col1, row2_col2 = st.columns(2)

# BOT-LEFT quarter: testing with examples
with row2_col1:
    st.header("Examples")

    subcol1, subcol2 = st.columns(2)

    with subcol1:
        # select number of examples
        n_examples = st.number_input("Number of Examples", 3)

    with subcol2:
        # seed next to number of examples
        seed = st.number_input("Seed", 42)

    # button to sample examples
    if st.button("Sample Examples"):
        random.seed(seed)
        examples = random.sample(lines, n_examples)

        # display samples
        for example in examples:
            st.write(f"{example}")

# BOT-RIGHT quarter: model prediction for selected example
with row2_col2:
    st.header("Model predictions")
    
    if st.button("Predict"):
        st.write("Predictions will appear here.")