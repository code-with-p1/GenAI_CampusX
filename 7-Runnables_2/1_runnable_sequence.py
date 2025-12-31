from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

joke_prompt = PromptTemplate(
    template='Write a joke about {topic} in limited words.',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

output_parser = StrOutputParser()

joke_explain_prompt = PromptTemplate(
    template='Explain the following joke in limited words - {text}',
    input_variables=['text']
)

chain = RunnableSequence(joke_prompt, model, output_parser, joke_explain_prompt, model, output_parser)

print(chain.invoke({'topic':'AI'}))