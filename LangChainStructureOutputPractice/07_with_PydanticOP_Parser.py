from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# Load API key
load_dotenv()


# Define output structure
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    city: str = Field(description="City where the person lives")


# Create parser
parser = PydanticOutputParser(pydantic_object=Person)


# Prompt template
prompt = PromptTemplate(
    template="""
Generate information about a person.

{format_instructions}

Person: {person}
""",
    input_variables=["person"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# Create chain
chain = prompt | model | parser


# Run chain
result = chain.invoke(
    {
        "person": "Albert Einstein"
    }
)

print(result)
print(type(result))