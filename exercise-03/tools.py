from langchain_groq import ChatGroq
from dotenv import load_dotenv
from state import State

load_dotenv()

def llm(model: str):
    return ChatGroq(model=model)

def route_task(state: State):
    if state["category"] == "simple":
        return "simple"

    return "complex"