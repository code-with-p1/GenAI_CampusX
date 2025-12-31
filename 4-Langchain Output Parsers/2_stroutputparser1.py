from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
    temperature=0.1,
    max_new_tokens=512,
)
model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary with TWO input variables
template2 = PromptTemplate(
    template="""Write a {style} summary of the following text.
    - Maximum length: {max_words} words
    - Tone: {tone}
    - Language: {language}
    Text:\n{text}""",
    input_variables=['style', 'max_words', 'tone', 'language', 'text']
)

parser = StrOutputParser()

# The chain now passes the output of the first model call as 'text' to the second prompt
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({
    'topic': 'Black Hole',
    'style': 'Academic',
    'max_words': 100,
    'tone': 'Formal',
    'language': 'English'
})

print(result)