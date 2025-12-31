from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create a dynamic prompt selector
def get_system_prompt(conversation_context):
    """Return appropriate system prompt based on context"""
    
    if "code" in conversation_context.lower():
        return """You are an expert programmer. 
        Provide code examples with explanations.
        Format code in markdown with proper syntax highlighting."""
    
    elif "math" in conversation_context.lower():
        return """You are a math tutor. 
        Explain concepts step by step.
        Use formulas and show calculations."""
    
    elif "creative" in conversation_context.lower():
        return """You are a creative writer.
        Be imaginative and descriptive.
        Use metaphors and vivid language."""
    
    else:
        return "You are a helpful AI assistant. Provide clear and accurate information."

# Chain with dynamic system prompt
def create_conversation_chain(user_input, chat_history):
    # Analyze conversation to determine context
    conversation_text = " ".join([msg.content for msg in chat_history[-3:]]) + " " + user_input
    
    # Get dynamic system prompt
    system_prompt = get_system_prompt(conversation_text)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Create chain
    chain = prompt | model | StrOutputParser()
    
    return chain.invoke({"input": user_input})

# Usage
chat_history = []
while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        break
    
    response = create_conversation_chain(user_input, chat_history)
    
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    
    print(f"AI: {response}")