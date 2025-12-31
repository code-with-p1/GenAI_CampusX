from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from transformers import logging
from dotenv import load_dotenv
import warnings
import os

load_dotenv()

os.environ["HF_HOME"] = r"G:\huggingface-cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def load_model(type, source=''):
    if type.lower == 'api':
        if source.lower == 'google':
            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        elif source.lower == 'huggingface':
            llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.2-1B-Instruct",
                task="text-generation",
                temperature=0,  # Slight increase for variation (optional)
                max_new_tokens=100,  # Ample for complete answers
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.15,
                return_full_text=True,  # Key fix: Returns only generation, no truncation
            )
            model = ChatHuggingFace(llm=llm)
        else:
            llm = HuggingFacePipeline.from_model_id(
                model_id="meta-llama/Llama-3.2-1B-Instruct",
                task="text-generation",
                pipeline_kwargs=dict(
                    temperature=0.5,
                    max_new_tokens=100
                )
            )
            model = ChatHuggingFace(llm=llm)
    else:
        llm = HuggingFacePipeline.from_model_id(
            model_id="meta-llama/Llama-3.2-1B-Instruct",
            task="text-generation",
            pipeline_kwargs=dict(
                temperature=0.5,
                max_new_tokens=100
            )
        )
        model = ChatHuggingFace(llm=llm)
    return model