from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load API key from .env
load_dotenv()

# Create Gemini chat model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# Create messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is Deep Learning?")
]

# Send messages to model
response = model.invoke(messages)

# Print AI response
print(response.content)