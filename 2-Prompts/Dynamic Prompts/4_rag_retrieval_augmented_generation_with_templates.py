from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Load and process documents
def create_knowledge_base():
    # Load documents (example with web content)
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore.as_retriever()

# Create RAG prompt template
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Use the following context to answer the question. 
    If you don't know the answer, say you don't know.
    
    Context: {context}
    """),
    ("human", "Question: {question}")
])

# Initialize retriever
retriever = create_knowledge_base()

# RAG chain
def rag_chain(question):
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Format prompt with context
    messages = rag_prompt.format_messages(context=context, question=question)
    
    # Get response
    result = model.invoke(messages)
    return result.content

# Usage
while True:
    user_input = input("You (ask about AI): ")
    
    if user_input.lower() == 'exit':
        break
    
    response = rag_chain(user_input)
    print(f"AI: {response}")