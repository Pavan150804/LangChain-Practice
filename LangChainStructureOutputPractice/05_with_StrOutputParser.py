from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key
load_dotenv()

# Create model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# Create prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)

# Create parser
parser = StrOutputParser()

# Create chain
chain = prompt | model | parser

# Run chain
response = chain.invoke(
    {
        "topic": "Machine Learning"
    }
)

print(response)
print(type(response))