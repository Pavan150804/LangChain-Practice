from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
# Load environment variables from .env
load_dotenv()

# Create OpenAI chat model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1.5,
    max_completion_tokens=10,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Send prompt to model
result = model.invoke("Write a story about Newton")

# Print only the generated text
print(result.content)