from state import State
from langchain.messages import AIMessage


def node3(state: State):
    
    return {"messages" : [AIMessage(content="The task is too complex, and I will not do it")], 
            "response":"The task is too complex, and I will not do it" 
            }
