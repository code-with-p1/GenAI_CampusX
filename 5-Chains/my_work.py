import sys
from pathlib import Path

# Add the parent directory (project_root) to Python's path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from common import load_model

load_dotenv()

model = load_model("Local")

template = PromptTemplate(
    template="Summerize the information in topic : {topic} in 5 lines.",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({'topic':'Cricket'})

print(result)