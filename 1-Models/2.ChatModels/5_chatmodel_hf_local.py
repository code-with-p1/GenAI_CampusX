from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import logging
import warnings
import os
import torch

load_dotenv()
# login()


os.environ["HF_HOME"] = r"G:\huggingface-cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is AI?")
print(result.content)