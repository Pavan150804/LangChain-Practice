from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Documents to convert into vectors
documents = [
    "I am good at Agentic AI",
    "He is very happy with outcome",
    "She went to meet her friends"
]

# Generate embeddings for all documents
vectors = embedding.embed_documents(documents)

# Print vectors
print(vectors)

# Number of documents
print("Number of vectors:", len(vectors))

# Dimension of each vector
print("Vector dimension:", len(vectors[0]))