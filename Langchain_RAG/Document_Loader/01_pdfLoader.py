#pip install pymupdf langchain-community
from langchain_community.document_loaders import PyMuPDFLoader

# Initialize the loader with the target PDF path
loader = PyMuPDFLoader("path/to/document.pdf")

# Load pages into a list of Document objects
docs = loader.load()

print(docs[0])