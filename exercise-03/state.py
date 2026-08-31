from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from pydantic import BaseModel

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    category: Literal["simple" , "complex"] 
    response: str


class Classification(BaseModel):
    category: Literal["simple" , "complex"] 
    notes: str = ""

class Response (BaseModel):
    response:str
    extra: str = ""