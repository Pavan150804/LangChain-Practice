from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# First Prompt
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

# Second Prompt
prompt2 = PromptTemplate(
    template="Generate a 5-point summary from the following text:\n{text}",
    input_variables=["text"]
)

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# Parser
parser = StrOutputParser()

# Sequential Chain
chain = (
    prompt1
    | model
    | parser
    | prompt2
    | model
    | parser
)

# Execute
result = chain.invoke(
    {
        "topic": "GPU usage in India"
    }
)

print(result)

chain.get_graph().print_ascii() #pip install grandalf