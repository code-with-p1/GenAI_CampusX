
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from common import load_model

model = load_model("Api", "Huggingface")

chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful AI assistant with expertise in {domain}. Always provide detailed explanations."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}")
])

chat_history = []

while True:

    user_input = input('You: ')
    
    if user_input == 'exit':
        break
    
    # Create dynamic prompt
    prompt = chat_prompt_template.format_messages(
        domain="Machine Learning",
        chat_history=chat_history,
        input=user_input
    )
    
    result = model.invoke(prompt)
    response = result.content
    
    # Update chat history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    
    print(f"AI: {response}")
    print("\n" + "="*50 + "\n")