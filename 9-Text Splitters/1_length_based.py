from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF
loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

# Splitter settings
splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    separator=''
)

# Split the documents
chunks = splitter.split_documents(docs)

# Proper output
print(f"Total number of chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks, start=1):
    print(f"--- Chunk {i} (page {chunk.metadata['page']}): ---")
    print(chunk.page_content.strip())
    print()