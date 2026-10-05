from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# Folder where Hugging Face downloads model files
os.environ["HF_HOME"] = r"D:\LangChainModelsPractice\huggingface_cache"

# Load TinyLlama model locally
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100
    }
)

# Convert LLM into a Chat Model
model = ChatHuggingFace(llm=llm)

# Send prompt
result = model.invoke("How are you doing today?")

# Print response
print(result.content)