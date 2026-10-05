from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Create model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# JSON Schema
person_schema = {
    "title": "Person",
    "description": "Information about a person",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Person's name"
        },
        "age": {
            "type": "integer",
            "description": "Person's age"
        },
        "city": {
            "type": "string",
            "description": "City where the person lives"
        }
    },
    "required": ["name", "age", "city"]
}

# Structured model
structured_model = model.with_structured_output(person_schema)

# Invoke model
result = structured_model.invoke(
    "Rahul is 25 years old and lives in Mumbai."
)

print(result)
print(type(result))