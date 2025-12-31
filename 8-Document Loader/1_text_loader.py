from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(type(docs))
print("="*80)
print(len(docs))
print("="*80)
print(docs[0].page_content)
print("="*80)
print(docs[0].metadata)
print("="*80)

prompt = PromptTemplate(
    template='Write a summary for the following poem in two lines only. - \n {poem}',
    input_variables=['poem']
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))

