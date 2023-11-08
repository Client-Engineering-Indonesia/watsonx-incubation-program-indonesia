import logging
import os
import pickle
import tempfile
import textwrap

try:
    from langchain import PromptTemplate
    from langchain.chains import RetrievalQAWithSourcesChain
    from langchain.memory import ConversationBufferMemory
except ImportError:
    raise ImportError("Could not import langchain: Please install ibm-generative-ai[langchain] extension.")

import streamlit as st
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import (HuggingFaceHubEmbeddings,
                                  HuggingFaceInstructEmbeddings)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from PIL import Image
from googletrans import Translator
import requests

from langChainInterface import LangChainInterface

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

    st.write("Configure model and parameters:")

    model_option = st.selectbox("Model Selected:", ["llama2-70b", "flan-ul2", "granite-13b"])
    chain_option = st.selectbox("Chain Type:", ["stuff", "refine", "mapReduce", "custom"])
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

st.markdown('<div style="text-align: right;">Powered by <span style="color: darkblue;">watsonx.ai</span></div>', unsafe_allow_html=True)

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
elif chain_option == "mapReduce":
    chain_types = "map_reduce"
else:
    chain_types = "custom"

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

system_prompt = """
    You are communicating with Vella, a knowledgeable, respectful, and precise assistant. My purpose is to provide accurate answers based on the information found in the document you provided.

    Please adhere to the following guidelines:
    1. Feel free to ask questions related to the document and I'll strive to deliver precise and relevant information.
    2. If your question doesn't directly pertain to the document's content, I'll search our conversation history for a relevant response. If none is found, I'll say "Please to ask me more detail".
    3. If a question's answer is irrelevant or unclear, I'll respond with "I don't know" or request clarification.
    4. For quick yes/no inquiries related to permissions, I'll provide concise responses along with document references.
    5. To maintain context in follow-up questions, I will explicitly reference the previous question, using clear language such as "it" to ensure coherent responses.
    6. When answering questions about numbered or ordered lists, I'll use line breaks for clarity and follow any specific ordering requests.
    7. To avoid redundancy, I'll keep track of your questions and provide concise, relevant answers. Feel free to ask follow-up questions for additional information.
    8. If you wish to engage in casual conversation, simply say "Hi" and I will respond your name with a friendly greeting.

    Please ensure that your responses are clear and respectful. If you have any issues or need clarification, don't hesitate to ask.
"""

def get_prompt_template(system_prompt=system_prompt, history=False):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    document_with_metadata_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="\nDocument: {page_content}\n\t",
    )
    
    if history:
        instruction = """
        Context: {history} \n {summaries}
        User question: {question}
        Answer the question in Markdown format,
        Markdown: """

        prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
        prompt = PromptTemplate(input_variables=["history", "summaries", "question"], template=prompt_template)
    else:
        instruction = """
        Context: {summaries}
        User: {question}
        Markdown:
        """

        prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
        prompt = PromptTemplate(input_variables=["summaries", "question"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return (
        document_with_metadata_prompt,
        prompt,
        memory,
    )

def setup_qa_chain(model_llm, db, system_prompt):
    retriever = db.as_retriever(search_kwargs={'k': 3})
    use_history = True

    doc_promt, prompt, memory = get_prompt_template(system_prompt=system_prompt, history=use_history)

    if use_history:
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=model_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt, "document_prompt": doc_promt, "memory": memory},
        )
    else:
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=model_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt, "document_prompt": doc_promt},
        )
    return qa

def translate_large_text(text, translate_function, choice, max_length=500):
    """
    Break down large text, translate each part, and merge the results.
    :param text: str, The large body of text to translate.
    :param translate_function: function, The translation function to use.
    :param max_length: int, The maximum character length each split of text should have.
    :return: str, The translated text.
    """
    
    # Split the text into parts of maximum allowed character length.
    text_parts = textwrap.wrap(text, max_length, break_long_words=True, replace_whitespace=False)

    translated_text_parts = []

    for part in text_parts:
        # Translate each part of the text.
        translated_part = translate_function(part, choice)  # Assuming 'False' is a necessary argument in the actual function.
        translated_text_parts.append(translated_part)

    # Combine the translated parts.
    full_translated_text = ' '.join(translated_text_parts)

    return full_translated_text


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

@st.cache_data
def read_pdf(uploaded_files, chunk_size=250, chunk_overlap=20):
    translated_docs = []

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
            # Write content to the temporary file
            temp_file.write(bytes_data)
            filepath = temp_file.name

            with st.spinner('Waiting for the file to upload'):
                loader = PyPDFLoader(filepath)
                data = loader.load()

                for doc in data:
                    # Extract the content of the document
                    content = doc.page_content

                    # Translate the content
                    translated_content = translate_large_text(content, translate_to_bahasa, False)

                    # Replace original content with translated content
                    doc.page_content = translated_content
                    translated_docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(translated_docs)
    
    return docs

def read_push_embeddings(docs):

    from utils.model import hugfacelib
    # repo_id="sentence-transformers/all-MiniLM-L6-v2"
    repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding = hugfacelib(repo_id)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding
        )
    
    # vectorstore = FAISS.from_documents(
    #     documents=docs,
    #     embedding=embedding,
    #     )
        
    return vectorstore

uploaded_file = st.file_uploader("Choose a PDF file", accept_multiple_files=True, type=["pdf"])

docs = read_pdf(uploaded_file)

if docs:
    db = read_push_embeddings(docs)
    st.write("\n")
    
else:
    st.error("Silahkan untuk unggah dokumen Anda")

st.markdown('<hr style="border: 1px solid #f0f2f6;">', unsafe_allow_html=True)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Halo, Silahkan tanya apa saja terkait dokumen Anda!"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
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

            try:
                db = db

                model_llm = LangChainInterface(model=model_id, credentials=creds, params=params, project_id=project_id)

                if chain_types == "custom":
                    chain = setup_qa_chain(model_llm, db, system_prompt)
                    res = chain(translated_user_input, return_only_outputs=True)
                    response = res['answer']

                else:
                    docs_search = db.similarity_search(translated_user_input, k=3)           
                    print(f"{docs_search}\n")

                    chain = load_qa_chain(model_llm, chain_type=chain_types)
                    response = chain.run(input_documents=docs_search, question=translated_user_input)

                if "<|endoftext|>" in response:
                    response = response.replace("<|endoftext|>", "")

                response = translate_to_bahasa(response, True)
                print(f"{response}\n")

            except NameError:
                response = "Silahkan untuk unggah dokumen Anda terlebih dahulu."

            placeholder = st.empty()
            full_response = ''

            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
