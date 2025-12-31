from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Chains Approach

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()

# Traditional Approach

from langchain_core.messages import HumanMessage

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

# Manual invocation without chain
formatted_prompt = prompt.format(topic='cricket')
message = HumanMessage(content=formatted_prompt)
response = model.invoke([message])
result = parser.invoke(response)

print(result)

# Note: Without a chain, there's no graph to print, so this line is omitted
