from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    temperature=0,  # Slight increase for variation (optional)
    max_new_tokens=100,  # Ample for complete answers
    do_sample=True,
    top_p=0.95,
    repetition_penalty=1.15,
    return_full_text=True,  # Key fix: Returns only generation, no truncation
)

model = ChatHuggingFace(llm=llm)
result = model.invoke([HumanMessage(content="What is AI?")])
print(result.content)