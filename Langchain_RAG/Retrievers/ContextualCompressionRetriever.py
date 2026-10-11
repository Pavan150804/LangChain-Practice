# Contextual Compression = Reducing retrieved documents to only the parts relevant to the current query.

# Instead of: Retrieve 5 full documents (each 1,000 tokens)

# It does:

#     Retrieve 5 documents
#     Then extract only the relevant sentences/paragraphs
#     Send compressed version to LLM


# New imports for LangChain >= 1.0
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever          # moved here
from langchain.retrievers.document_compressors import LLMChainExtractor  # may still work
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# Create a FAISS vector store from the documents
vectorstore = FAISS.from_documents(
    docs, 
    embedding_model
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Set up the compressor using an LLM
llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")
compressor = LLMChainExtractor.from_llm(llm)

# Create the contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Query the retriever
query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)


for i, doc in enumerate(compressed_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)