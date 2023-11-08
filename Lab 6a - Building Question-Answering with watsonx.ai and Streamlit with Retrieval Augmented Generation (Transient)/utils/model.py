import os
import dotenv
dotenv.load_dotenv()

### Load the credentials
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)

min_new_tokens=1
max_new_tokens=300
temperature=1
top_k=50
top_p=1
random_seed=42
repetition_penalty=1.2


from langchain.callbacks.base import BaseCallbackHandler
class StreamHandler(BaseCallbackHandler):

    from uuid import UUID
    from langchain.schema.output import LLMResult
    from typing import Any

    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            self.text += token
            self.container.markdown(self.text)
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    

### IBM Watson
def set_params_ibm(decoding_method):
    from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

    if decoding_method == "sample":
        params = {
            GenParams.DECODING_METHOD: decoding_method,
            GenParams.MIN_NEW_TOKENS: min_new_tokens,
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.TEMPERATURE: temperature,
            GenParams.TOP_K: top_k,
            GenParams.TOP_P: top_p,
            GenParams.RANDOM_SEED: random_seed,
            GenParams.REPETITION_PENALTY: repetition_penalty,
            GenParams.STOP_SEQUENCES: ["# [","\n\n","**"],
        }
    elif decoding_method == "greedy":
        params = {
            GenParams.DECODING_METHOD: decoding_method,
            GenParams.MIN_NEW_TOKENS: min_new_tokens,
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.RANDOM_SEED: random_seed,
            GenParams.TEMPERATURE: temperature,
            GenParams.REPETITION_PENALTY: repetition_penalty,
            GenParams.STOP_SEQUENCES: ["#","[","\n\n","**"," * "],
        }
    else:
        params = None
        print("Decoding method not supported, please use 'sample' or 'greedy'.")
    
    return params

### IBM Research BAM
def llm_param(decoding_method, model_id):

    from genai.credentials import Credentials
    from genai.model import Model
    from genai.extensions.langchain import LangChainInterface

    if bam_api_key is None or bam_api_url is None:

        llm = None

        print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")

    else:
        creds = Credentials(bam_api_key, api_endpoint=bam_api_url)
        
        # decoding_method = "greedy"
        params = set_params_ibm(decoding_method)
 
        message_placeholder = None  # You can define this if needed
        stream_handler = StreamHandler(message_placeholder)

        callback = False
        if callback is True:
            llm = LangChainInterface(
                model=model_id,
                credentials=creds,
                params=params,
                callbacks=[stream_handler]
            )
        else:
            llm = LangChainInterface(
                model=model_id,
                credentials=creds,
                params=params
            )

        llm = LangChainInterface(model=model_id, credentials=creds, params=params, project_id=project_id)

    return llm

### Hugging Face
def hugfacelib(repo_id):

    from langchain.embeddings import HuggingFaceHubEmbeddings

    repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding = HuggingFaceHubEmbeddings(
        task="feature-extraction",
        repo_id = repo_id,
        huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN
        )

    return embedding
