from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Dynamic template with multiple variables
def create_personalized_prompt(user_info, current_topic, history):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        User Profile:
        Name: {name}
        Expertise Level: {expertise}
        Interests: {interests}
        
        Current Date: {date}
        Conversation Topic: {topic}
        
        Previous messages: {history}
        
        Tailor your response to match the user's expertise level and interests.
        """),
        ("human", "{message}")
    ])
    
    return prompt.format_messages(
        name=user_info.get("name", "User"),
        expertise=user_info.get("expertise", "Beginner"),
        interests=", ".join(user_info.get("interests", [])),
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        topic=current_topic,
        history=history[-5:] if len(history) > 5 else history,
        message=user_input
    )

# User profile (can be loaded from database/API)
user_profile = {
    "name": "Alex",
    "expertise": "Intermediate",
    "interests": ["AI", "Python", "Data Science"]
}

chat_history = []

while True:
    user_input = input("You: ")
    
    if user_input == 'exit':
        break
    
    # Analyze topic from user input
    if any(word in user_input.lower() for word in ["code", "program", "function"]):
        topic = "Programming"
    elif any(word in user_input.lower() for word in ["data", "analysis", "statistics"]):
        topic = "Data Science"
    else:
        topic = "General"
    
    # Create dynamic prompt
    messages = create_personalized_prompt(
        user_profile, 
        topic, 
        chat_history
    )
    
    result = model.invoke(messages)
    response = result.content
    
    # Update history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    
    print(f"AI: {response}")