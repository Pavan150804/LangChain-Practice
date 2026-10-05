from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

parser = StrOutputParser()

# Summary Prompt
summary_prompt = PromptTemplate(
    template="Give a short summary of {topic}",
    input_variables=["topic"]
)

# Advantages Prompt
advantages_prompt = PromptTemplate(
    template="What are the advantages of {topic}?",
    input_variables=["topic"]
)

# Parallel Chain
parallel_chain = RunnableParallel(
    summary=summary_prompt | model | parser,
    advantages=advantages_prompt | model | parser
)

# Execute
result = parallel_chain.invoke(
    {
        "topic": "Deep Learning"
    }
)

print(result)