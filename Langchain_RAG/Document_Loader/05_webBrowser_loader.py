from langchain_community.document_loaders import WebBaseLoader

# Website URL
loader = WebBaseLoader(
    "https://en.wikipedia.org/wiki/Artificial_intelligence"
)

# Load page content
docs = loader.load()

print("Number of documents:", len(docs))
print("\nPage Content:\n")
print(docs[0].page_content[:1000])  # first 1000 characters