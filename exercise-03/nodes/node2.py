from state import State, Response
from langchain.messages import AIMessage , SystemMessage
from tools import llm

def node2(state: State):

    system_message = SystemMessage(
        content=(
            "You are the simple-task response agent. "
            "Answer ONLY simple user requests. "
            "Provide a concise and direct answer. "
            "Do not attempt to solve complex tasks. "
            "Do not perform multi-step analysis or extensive reasoning. "
            "If the conversation contains a complex request, do not solve it."
        )
    )

    messages = [system_message, *state["messages"]]

    chat = llm("openai/gpt-oss-120b")
    #structure_chat = chat.with_structured_output(Response)
    result = chat.invoke(messages)

    return{
        "messages": [result],
        "response": result.content,
    }
