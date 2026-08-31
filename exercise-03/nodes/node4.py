from state import State
from langchain_core.messages import RemoveMessage


def node4(state: State):
    messages = state["messages"]

    if len(messages) <= 9:
        return {}

    return {
        "messages": [
            RemoveMessage(id=m.id) for m in messages[:-2]
        ]
    }