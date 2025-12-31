from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

joke_prompt = PromptTemplate(
    template='Write a single joke about {topic} in limited words.',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

joke_explain_prompt = PromptTemplate(
    template='Explain the following joke in limited words - {text}',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(joke_prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(joke_explain_prompt, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'cricket'}))