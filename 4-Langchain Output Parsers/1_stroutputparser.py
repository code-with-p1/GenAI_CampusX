from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from transformers import logging
import warnings
import os

load_dotenv()
os.environ["HF_HOME"] = r"G:\huggingface-cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.1,  # Lower temp for more deterministic output
    max_new_tokens=512,  # Limit to avoid rambling
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})

result = model.invoke(prompt1)

print("Detailed Report : ", result.content)

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)

print("\n\n")

print("Summary : ", result1.content)