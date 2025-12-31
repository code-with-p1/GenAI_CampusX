from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
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
    temperature=1.5,  # Lower temp for more deterministic output
    max_new_tokens=512,  # Limit to avoid rambling
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place':'Germany'})

print(f"Name : {final_result.name}")
print(f"Age : {final_result.age}")
print(f"City : {final_result.city}")