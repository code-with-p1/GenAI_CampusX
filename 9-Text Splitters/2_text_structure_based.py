from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.
These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# Initialize the splitter with good default settings
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 250,
    chunk_overlap = 25,
    separators=["\n\n", "\n", " ", ""],
    strip_whitespace=True
)

# Perform the split on raw text
chunks = splitter.split_text(text)

# Proper, readable output
print(f"Total number of chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks, start=1):
    print(f"--- Chunk {i} ({len(chunk)} characters) ---")
    print(chunk.strip())
    print()