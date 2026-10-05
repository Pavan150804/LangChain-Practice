from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# Prompt Template
prompt = PromptTemplate(
    template="""
Write a short summary for the following story:

{story}
""",
    input_variables=["story"]
)

# Output Parser
parser = StrOutputParser()

# Load Text File
loader = TextLoader("movie.txt", encoding="utf-8")

docs = loader.load()

# Extract Text
story_text = docs[0].page_content

# Create Chain
chain = prompt | model | parser

# Run Chain
result = chain.invoke(
    {
        "story": story_text
    }
)

print("\nSummary:\n")
print(result)