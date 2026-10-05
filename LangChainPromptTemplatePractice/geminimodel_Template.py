# pip install langchain-core
# pip install langchain_google_genai

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

prompt_temp=PromptTemplate(
    input_variables=["topic","word_count"],
    template="can you explain the topic of {topic} of {word_count} words" ,
    validate_template=True 
)

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash",api_key="")

actual_prompt=prompt_temp.invoke(
    {
        "topic": "deep learning",
        "word_count": 300
    }
)

response=model.invoke(actual_prompt)




