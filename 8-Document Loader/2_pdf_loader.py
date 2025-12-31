from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(type(docs))
print("="*80)
print(len(docs))
print("="*80)
print(docs[0].page_content)
print("="*80)
print(docs[1].page_content)
print("="*80)
print(docs[0].metadata)
print("="*80)
print(docs[0].metadata['total_pages'])
print("="*80)

for page_no in range(docs[0].metadata['total_pages']):
    print(f"Page No {page_no+1}")
    print(docs[page_no].page_content)
    print("="*80)