
# Import environment loading library
from dotenv import load_dotenv
# Import IBMGen Library 
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain.llms.base import LLM
# Import lang Chain Interface object
from langChainInterface import LangChainInterface
# Import system libraries
import os
# Import streamlit for the UI 
import streamlit as st
import re
# import qa-model
from qa_model import *

# Load environment vars
load_dotenv()

# Define credentials 
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)


if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }

# Define generation parameters 
params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 600,
    GenParams.TEMPERATURE: 0,
    # GenParams.TOP_K: 100,
    # GenParams.TOP_P: 1,
    GenParams.REPETITION_PENALTY: 1
    #GenParams.STOP_SEQUENCES: ["Pertanyaan"]
}


granite_chat ="ibm/granite-13b-chat-v1"
flanul= "google/flan-ul2"
llama2= "meta-llama/llama-2-70b-chat"
mt_zero = "bigscience/mt0-xxl-13b"

llm = LangChainInterface(model=llama2, credentials=creds, params=params, project_id=project_id)

# Title for the app
st.title('🤖 Model AI untuk Tanya Jawab')

# Prompt box
prompts = st.text_input('Masukan Pertanyaan Anda Disini')

# Submit button
if st.button('Submit'):
    if prompts:
        # Show loading spinner
        with st.spinner('Memproses...'):
            answer_generations = answer_retrieval(llm, question_prompt(user_question=prompts, qa_sample=question_answer_id))
            st.write(answer_generations)
    else:
        st.warning('Masukan Prompt Anda Terlebih Dahulu.')

