from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

# Load API Key
load_dotenv()

# Gemini Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# String Parser
parser = StrOutputParser()

# Structured Output Schema
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )

# Pydantic Parser
parser2 = PydanticOutputParser(
    pydantic_object=Feedback
)

# Sentiment Classification Prompt
prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback text into positive or negative.

Review:
{review}

{parser_instruction}
""",
    input_variables=["review"],
    partial_variables={
        "parser_instruction": parser2.get_format_instructions()
    }
)

# Classifier Chain
classifier_chain = prompt1 | model | parser2

# Positive Response Prompt
prompt2 = PromptTemplate(
    template="""
Write an appropriate response to this positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)

# Negative Response Prompt
prompt3 = PromptTemplate(
    template="""
Write an appropriate response to this negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)

# Conditional Branch
branch_chain = RunnableBranch(

    (
        lambda x: x.sentiment == "positive",
        prompt2 | model | parser
    ),

    (
        lambda x: x.sentiment == "negative",
        prompt3 | model | parser
    ),

    RunnableLambda(
        lambda x: "Could not determine sentiment"
    )
)

# Complete Conditional Chain
chain = classifier_chain | branch_chain

# Execute
result = chain.invoke(
    {
        "review": "This is a bad phone"
    }
)

print(result)

# Optional: Display Graph
chain.get_graph().print_ascii()