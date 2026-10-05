from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Load API key
load_dotenv()

# Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# Output Parser
parser = StrOutputParser()

# Simple Chain
chain = prompt | model | parser

# Execute
result = chain.invoke(
    {
        "topic": "Machine Learning"
    }
)

print(result)