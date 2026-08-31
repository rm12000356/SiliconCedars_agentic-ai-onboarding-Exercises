from langchain_groq import ChatGroq
from state import State

def llm(model: str):
    return ChatGroq(model=model)

def route_task(state: State):
    if state["category"] == "simple":
        return "simple"

    return "complex"