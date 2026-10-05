import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

st.title("AI Chatbot")

question = st.text_input("Ask a question") #st.text_input() is a Streamlit widget that creates a text box where the user can type input.

if question:
    response = model.invoke(question)
    st.write(response.content)