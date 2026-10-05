from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=""
)

class Person(TypedDict):
    name: str
    age: int
    DOB: str

structured_model = model.with_structured_output(Person)

result = structured_model.invoke(
    """
    Sachin Tendulkar is 52 years old as of November 2025.
    He was born on April 24, 1973.
    """
)
print(type(result)) # dict
print(result)