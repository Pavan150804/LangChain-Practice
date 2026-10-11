from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load API Key from .env
load_dotenv()

# Create Documents
doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history and led Mumbai Indians to five titles.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni has led Chennai Super Kings to multiple IPL titles and is known as Captain Cool.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is one of the best fast bowlers in T20 cricket.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a world-class all-rounder and key player for Chennai Super Kings.",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

# Gemini Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# Create Chroma Vector Store
vector_store = Chroma(
    collection_name="ipl_players",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Store Documents
#vector_store.add_documents(docs)

print("Documents stored successfully!")

# view documents
# print(vector_store.get(include=['embeddings','documents', 'metadatas']))


# search documents
# print(vector_store.similarity_search(
#     query='Who among these are a bowler?',
#     k=2
# ))


# search with similarity score
# vector_store.similarity_search_with_score(
#     query='Who among these are a bowler?',
#     k=2
# )

# meta-data filtering
print(vector_store.similarity_search_with_score(
    query="who is the all rounder in csk",
    filter={"team": "Chennai Super Kings"}
))

# update documents
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)
vector_store.update_document(document_id='09a39dc6-3ba6-4ea7-927e-fdda591da5e4', document=updated_doc1)

# view documents
vector_store.get(include=['embeddings','documents', 'metadatas'])

# delete document
vector_store.delete(ids=['09a39dc6-3ba6-4ea7-927e-fdda591da5e4'])

# view documents
vector_store.get(include=['embeddings','documents', 'metadatas']) 