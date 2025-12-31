from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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
parser = JsonOutputParser()

# Enhanced template with stricter instructions
template = PromptTemplate(
    template="""You are a helpful assistant that outputs ONLY valid JSON. Do not include any Markdown like ```json.
        Ensure every JSON object has commas between keys and values.
        Give me exactly 5 facts about {topic}. For each fact, provide a short description.
        {format_instruction}
    """,
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic': 'black hole'})
print(result)