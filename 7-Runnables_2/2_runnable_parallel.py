from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

tweet_prompt = PromptTemplate(
    template='Generate a tweet about {topic} with lesser words.',
    input_variables=['topic']
)

linkedin_prompt = PromptTemplate(
    template='Generate a Linkedin post about {topic} with lesser words.',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

output_parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(tweet_prompt, model, output_parser),
    'linkedin': RunnableSequence(linkedin_prompt, model, output_parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linkedin'])

