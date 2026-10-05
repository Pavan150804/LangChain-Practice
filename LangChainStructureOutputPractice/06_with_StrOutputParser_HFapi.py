from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint (
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

chat_model=ChatHuggingFace(llm=llm)

template1=PromptTemplate.from_template('write a detailed report on {topic}')

chain=template1 | chat_model
report=chain.invoke({"topic":"black hole"})

print("Detailed Report:\n",report)

str_output=StrOutputParser()
chain = template1 | chat_model | str_output
report=chain.invoke({"topic":"black hole"})

print("Detailed Report:\n",report)
