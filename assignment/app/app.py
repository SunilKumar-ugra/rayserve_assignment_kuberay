#app.py
import streamlit as st
import requests

# FastAPI endpoint
import os
API_URL = os.getenv("API_URL", "http://localhost:8000/")


# Function to interact with FastAPI
def process_text_via_api(input_text):
    try:
        response = requests.post(API_URL, json={"text": input_text})
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Unknown error")}
    except Exception as e:
        return {"error": str(e)}


# Streamlit UI
st.title("Text Processing Pipeline")
st.write("Summarization, Sentiment Analysis, and LLM Response Generation")

# Input text
input_text = st.text_area("Enter your text:", height=200)

if st.button("Process Text"):
    if input_text.strip():
        with st.spinner("Processing..."):
            results = process_text_via_api(input_text)
        
        if "error" in results:
            st.error(f"Error: {results['error']}")
        else:
            st.success("Processing Complete!")
            st.subheader("Summary:")
            st.write(results["summary"])

            st.subheader("Sentiments:")
            st.write(results["sentiments"])

            st.subheader("LLM Response:")
            st.write(results["llm_response"]["response"])
    else:
        st.warning("Please enter some text.")


