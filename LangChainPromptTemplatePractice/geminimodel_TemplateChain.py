from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# prompt_temp=PromptTemplate(
#     input_variables=["topic","word_count"],
#     template="can you explain the topic of {topic} of {word_count} words" ,
#     validate_template=True 
# )

# model=ChatGoogleGenerativeAI(model="gemini-3.6-flash",api_key="")

# chain = prompt_temp | model

# response=chain.invoke(
#     {
#         "topic": "deep learning",
#         "word_count": 300
#     }
    
# )

# print(response.content)


# from langchain_core.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.output_parsers import StrOutputParser

prompt_temp = PromptTemplate(
    input_variables=["topic", "word_count"],
    template="Can you explain {topic} in {word_count} words?"
)

prompt_temp2 = PromptTemplate(
    input_variables=["content"],
    template="Summarize the following content:\n\n{content}"
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=""
)

parser = StrOutputParser()

chain = (
    prompt_temp
    | model
    | parser
    | prompt_temp2
    | model
    | parser
)

response = chain.invoke(
    {
        "topic": "deep learning",
        "word_count": 300
    }
)

print(response)
