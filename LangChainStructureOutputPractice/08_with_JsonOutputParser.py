from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import JsonOutputParser


# Load API Key
load_dotenv()


# Create JSON Parser
parser = JsonOutputParser()

# Prompt Template
prompt = PromptTemplate(
    template="""
Generate details about a person.

Return the response in JSON format with the following fields:
name
age
city

Person: {person}
""",
    input_variables=["person"]
)

# Gemini Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# Chain
chain = prompt | model | parser

# Invoke
result = chain.invoke(
    {
        "person": "Albert Einstein"
    }
)

print(result)
print(type(result))