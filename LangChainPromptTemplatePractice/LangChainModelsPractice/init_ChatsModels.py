import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

#model=chatGoogleGenerativeAI(model=os.getenv("model_name"))

model = init_chat_model(os.getenv("model_name"))

response=model.invoke("what is AI")

print(response.content)