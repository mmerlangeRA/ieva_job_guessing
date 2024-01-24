import streamlit as st
from openai import OpenAI
import pandas as pd
import json
from io import BytesIO

api_key = "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"
temperature=0.0
max_tokens=1000
frequency_penalty=0.0


client = OpenAI(api_key=api_key)

def read_file(file):
    return file.read().decode("utf-8")

def read_prompt_template(file_path):
    with open(file_path, "r") as file:
        return file.read()
    
prompt_template = read_prompt_template("prompt_template.txt")

def parse_json(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        st.error("Invalid JSON format")
        return None

def specific_json_format(input_json):
    returned_json = {}
    for key, value in input_json.items():
        returned_json[key] = input_json[key]["assessed"]
        returned_json[key+"_details"] = input_json[key]["details"]
    return returned_json

def json_to_df(json_data):
    rows = []
    for key, value in json_data.items():
        row = {"Category": key, "Assessed": value["assessed"], "Details": value["details"]}
        rows.append(row)
    return pd.DataFrame(rows)

def call_openai_api(model_choice, text_content):
    try:
       
        message=[{"role": "user", "content": text_content}]
        response = client.chat.completions.create(model=model_choice,
                messages = message,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,  # Number of responses
                stop=None)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"


def process_files(files, model_choice):
    results = []
    progress_bar = st.progress(0)
    index=0
    total_files = len(files)
    for file in files:
        file_content = read_file(file)
        prompt = prompt_template.format(file_content=file_content)
        result = call_openai_api(model_choice, prompt)
        json_data = parse_json(result)
        results.append(specific_json_format(json_data))
        index+=1
        progress_bar.progress((index) / total_files)
        df = json_to_df(json_data)
        st.table(df)
    return results

# Streamlit app
def main_old():
    st.title("ChatGPT Text File Interaction")

    model_choice = st.selectbox("Choose a GPT model", ["gpt-3.5-turbo", "gpt-4"])
    
    # File upload widget
    uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        results = []
        for index, uploaded_file in enumerate(uploaded_files):
            file_content = read_file(uploaded_file)
            prompt = prompt_template.format(file_content=file_content)
            result = call_openai_api(model_choice, prompt)
            json_data = parse_json(result)
            results.append(specific_json_format(json_data))
            df = json_to_df(json_data)
            st.table(df)
            progress_bar.progress((index + 1) / total_files)
        if st.button("Process Files"):
            # Convert results to a DataFrame
            df = pd.DataFrame(results)            
            # Convert DataFrame to CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download results as CSV",
                data=BytesIO(csv),
                file_name="gpt_results.csv",
                mime="text/csv",
            )
def main():
    st.title("Guessing job data from description with Chatgpt")
    model_choice = st.selectbox("Choose a GPT model", ["gpt-3.5-turbo", "gpt-4"])
    
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = None

    uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)

    if uploaded_files and st.button("Process Files"):
        with st.spinner("Processing files..."):
            st.session_state['processed_data'] = process_files(uploaded_files, model_choice)

    if st.session_state['processed_data']:
        df = pd.DataFrame(st.session_state['processed_data'])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=BytesIO(csv),
            file_name="gpt_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
