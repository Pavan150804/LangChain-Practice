from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Create Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Ask a question
result = model.invoke("What is the purpose of Generative AI")

# Print response
print(result.content)