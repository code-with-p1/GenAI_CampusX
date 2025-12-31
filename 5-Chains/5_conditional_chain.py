from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

main_parser = PydanticOutputParser(pydantic_object=Feedback)

main_prompt = PromptTemplate(
    template="Give the output as positive or negative for following feedback. \n {feedback} {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': main_parser.get_format_instructions()}
)

common_parser = StrOutputParser()

positive_prompt = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

negative_prompt = PromptTemplate(
    template="Write a appropritate response to this negative feedback \n {feedback} ",
    input_variables=['feedback']
)

parallel_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive', positive_prompt | model | common_parser),
    (lambda x : x.sentiment == 'negative', negative_prompt | model | common_parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = main_prompt | model | main_parser | parallel_chain

feedback = 'This is the best phone ever.'

result = chain.invoke({'feedback':feedback})

print(result)
