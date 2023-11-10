import logging 
import os
import streamlit as st
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from PIL import Image
from googletrans import Translator

from langChainInterface import LangChainInterface

from pymilvus import (
    connections,
    Collection
)
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document

# Most GENAI logs are at Debug level.
# logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

load_dotenv()

handler = StdOutCallbackHandler()

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

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

header_text = 'RAG in Bahasa Indonesia <span style="color: blue; font-family: Cormorant Garamond; font-size: 40px;">| Watsonx</span>'
st.markdown(f'<h1 style="color: black;">{header_text}</h1>', unsafe_allow_html=True)

# Define model and chain type options
with st.sidebar:
    # st.title("Transient RAG Model in Bahasa")

    image = Image.open('watsonxai.jpg')
    # st.image(image, caption='Powered by watsonx.ai')
    st.image(image, caption='watsonx.ai, a next generation enterprise studio for AI builders to train, validate, tune and deploy AI models')
    # st.write("a next generation enterprise studio for AI builders to train, validate, tune and deploy AI models")
    st.write("Configure model and parameters:")

    model_option = st.selectbox("Model Selected:", ["llama2-70b", "flan-ul2", "granite-13b"])
    chain_option = st.selectbox("Chain Type:", ["stuff", "refine", "mapReduce"])
    decoding_option = st.selectbox("Decoding Parameter:", ["greedy", "sample"])
    max_new_tokens = st.number_input("Max Tokens:", 1, 1024, value=256)
    min_new_tokens = st.number_input("Min Tokens:", 0, value=8)
        
    st.markdown('''
    This app is an LLM-powered RAG built using:
    - [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai/)
    - [HuggingFace](https://huggingface.co/)
    - [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) [LLM model](https://python.langchain.com/docs/get_started/introduction)
    ''')
    # st.markdown('Powered by <span style="color: darkblue;">watsonx.ai</span>', unsafe_allow_html=True)

# st.markdown('<div style="text-align: right;">Powered by <span style="color: darkblue;">watsonx.ai</span></div>', unsafe_allow_html=True)
st.markdown('<hr style="border: 1px solid #f0f2f6;">', unsafe_allow_html=True)

decoding = "sample" if decoding_option == "Sample" else "greedy"

if model_option == "llama2-70b":
    model_id = ModelTypes.LLAMA_2_70B_CHAT.value
elif model_option == "flan-ul2":
    model_id = ModelTypes.FLAN_UL2.value
else:
    model_id = ModelTypes.GRANITE_13B_CHAT.value

if chain_option == "refine":
    chain_types = "refine"
elif chain_option == "stuff":
    chain_types = "stuff"
else:
    chain_types = "map_reduce"

if decoding == "greedy":

    params = {
            GenParams.DECODING_METHOD: decoding_option,
            GenParams.MIN_NEW_TOKENS: min_new_tokens,
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.TEMPERATURE: 1,
            GenParams.REPETITION_PENALTY: 1.2,
            GenParams.STOP_SEQUENCES: [" # ", " ** " ," * ", "<|endoftext|>", "--"],
        }
else:
        params = {
            GenParams.DECODING_METHOD: decoding_option,
            GenParams.MIN_NEW_TOKENS: min_new_tokens,
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.TEMPERATURE: 1,
            GenParams.TOP_K: 100,
            GenParams.TOP_P: 1,
            GenParams.REPETITION_PENALTY: 1.2,
            GenParams.STOP_SEQUENCES: [" # ", " ** " ," * ", "<|endoftext|>", "--"],
        }

def translate_to_bahasa(sentence: str, choice: bool) -> str:
    """
    Translate the text between English and Bahasa based on the 'choice' flag.
    
    Args:
        sentence (str): The text to translate.
        choice (bool): If True, translates text to Bahasa. If False, translates to English.
    Returns:
        str: The translated text.
    """
    translator = Translator()
    try:
        if choice:
            # Translate to Bahasa
            translate = translator.translate(sentence, dest='id')
        else:
            # Translate to English
            translate = translator.translate(sentence, dest='en')
        return translate.text
    except Exception as e:
        # Handle translation-related issues (e.g., network error, unexpected API response)
        raise ValueError(f"Translation failed: {str(e)}") from e

def milvus_search(query):
    model_name = 'all-MiniLM-L6-v2'
    collection_name = "WikiHow"
    # connections.connect("default", host="128.168.140.66", port="19530")
    connections.connect("default", host="localhost", port="19530")
    collection = Collection(collection_name)
    collection.load()
    search_params = {
        "metric_type": "L2",
        "params": {"ef": 10},
    }
    model = SentenceTransformer(model_name)
    vectors_to_search = model.encode([query]).tolist()

    result = collection.search(vectors_to_search, "vector", search_params,
                                limit=3,
                                output_fields=["text", "vector"],
                                )

    hits = result[0]
    def text2doc(t):
        return Document(page_content = t)

    docs = [text2doc(h.entity.get('text')) for h in hits]
    return docs


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Halo, Silahkan tanya apa saja terkait dokumen Anda!"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input("Send a message...", key="prompt"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    print(f"\n{user_question}\n")
    translated_user_input = translate_to_bahasa(user_question, False)
    print(f"{translated_user_input}\n")


if st.session_state.messages[-1]["role"] != "assistant":

    with st.chat_message("assistant"):
        with st.spinner("Harap Tunggu..."):

            docs_search = milvus_search(translated_user_input)
            print(f"{docs_search}\n")

            model_llm = LangChainInterface(model=model_id, credentials=creds, params=params, project_id=project_id)

            chain = load_qa_chain(model_llm, chain_type=chain_types)
            response = chain.run(input_documents=docs_search, question=translated_user_input)

            if "<|endoftext|>" in response:
                response = response.replace("<|endoftext|>", "")

            response = translate_to_bahasa(response, True)
            print(f"{response}\n")

            placeholder = st.empty()
            full_response = ''

            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
