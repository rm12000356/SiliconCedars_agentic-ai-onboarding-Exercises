from state import State 
from tools import llm
from langchain.messages import SystemMessage
from tools import calculator, web_search, document_lookup

def node1(state: State):

    SystemM = SystemMessage(
        content=(
            "You are a helpful assistant with access to three tools:\n"
            "- calculator: for math expressions\n"
            "- web_search: for current events or general knowledge not in local documents\n"
            "- document_lookup: for questions about internal company facts (vacation policy, "
            "remote work policy, product specs, support tiers)\n\n"
            "Before calling a tool, briefly reason about which tool (if any) is appropriate "
            "for the user's request. Only call a tool if it's actually needed to answer "
            "accurately; if you already know the answer with certainty, answer directly."
        )
    )
    message = [SystemM] + state["messages"]

    chat = llm("openai/gpt-oss-120b")
    chat_with_tools = chat.bind_tools([calculator, web_search, document_lookup])
    result = chat_with_tools.invoke(message)

    if result.tool_calls:
        for call in result.tool_calls:
            print(f"[reasoning] decided to call '{call['name']}' with args {call['args']}")
    else:
        print("[reasoning] no tool needed, answering directly")
    
    return{ "messages": [result] }