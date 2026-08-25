from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json
from analysis import analyze_results

load_dotenv()

end_results= []

class Person(BaseModel):
    name: str = Field( description="Full name of the person")
    age: int = Field( description="Age of the person")
    email: str = Field( description="Email address of the person")
    good_man: bool = Field( description="Indicates if the person is a good man")




def extract_person_info(text: str, text2: str, model_name: str) -> None:
    import time
    start = time.perf_counter()
    
    system_message = SystemMessage(
        content="You are a helpful assistant that extracts structured data from unstructured text."
    )

    human_message= "Extract the following information from the text: {text} {text2}."

    
    chat_prompt_system = ChatPromptTemplate.from_messages([system_message, human_message])
    formated_messages = chat_prompt_system.invoke({"text": text , "text2": text2})

    response_format ={
        "type": "json_schema",
        "json_schema": {
            "name": "Person",
            "schema": Person.model_json_schema()
        }
    }
    
    llm = ChatGroq(
        model=model_name,
        response_format=response_format
    )

    result = llm.invoke(formated_messages)
    content = result.content.strip()
    metadata = result.response_metadata
    token_usage = metadata.get("token_usage", {})

    end = time.perf_counter()
    elapsed = end - start

    end_results.append({
        "model": model_name,
        "time": elapsed,
        "content": content,
        "input_tokens": token_usage.get("prompt_tokens", 0),
        "output_tokens": token_usage.get("completion_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
        "reasoning_tokens": token_usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    })


text = "John Doe is a 30-year-old software engineer. You can reach him at john.doe@example.com or (555) 123-4567."
text2 = "he is a good man"

models = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-safeguard-20b",
]
for i in range(3):
    for model_name in models:
        extract_person_info(text, text2, model_name=model_name)

with open("results.json", "w") as file:
    json.dump(end_results, file, indent=4)

analyze_results("results.json")




