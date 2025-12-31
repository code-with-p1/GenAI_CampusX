from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create a Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}. Answer should be very short in length. Dont use more than 15 words."
)

# Define the input
topic = input('Enter a topic : ')

# Format the prompt manually using PromptTemplate
formatted_prompt = prompt.format(topic=topic)

# Call the LLM directly
blog_title = llm.invoke(formatted_prompt)

# Print the output
print("Generated Blog Title:", blog_title)