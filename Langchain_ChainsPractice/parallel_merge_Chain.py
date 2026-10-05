from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Load environment variables
load_dotenv()

# Gemini Model
model = ChatGoogleGenerativeAI( 
    model="gemini-3.6-flash"
)

# Output Parser
parser = StrOutputParser()

# Prompt 1 - Generate Notes
prompt1 = PromptTemplate(
    template="""
Generate short and simple notes from the following text:

{text}
""",
    input_variables=["text"]
)

# Prompt 2 - Generate Quiz
prompt2 = PromptTemplate(
    template="""
Generate 5 short question-answer pairs from the following text:

{text}
""",
    input_variables=["text"]
)

# Prompt 3 - Merge Results
prompt3 = PromptTemplate(
    template="""
Merge the following notes and quiz into one study document.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

# Parallel Execution
parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model | parser,
        "quiz": prompt2 | model | parser
    }
)

# Merge Chain
merge_chain = prompt3 | model | parser

# Complete Chain
chain = parallel_chain | merge_chain

# Input Text
text = """
Support Vector Machines (SVMs) are supervised machine learning algorithms.
They are commonly used for classification and regression tasks.
SVM works by finding the optimal hyperplane that separates different classes.
"""

# Run Chain
result = chain.invoke(
    {
        "text": text
    }
)

print(result)