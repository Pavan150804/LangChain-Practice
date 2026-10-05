from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()

# Create Chat Prompt Template
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain {topic} in simple terms.")
])

# Create Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# Create Chain
chain = chat_template | model

# Execute Chain
response = chain.invoke({
    "domain": "AI",
    "topic": "Deep Learning"
})

# Print response
print(response.content)