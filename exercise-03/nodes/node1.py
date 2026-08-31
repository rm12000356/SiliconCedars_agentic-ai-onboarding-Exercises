from state import State , Classification
from tools import llm
from langchain.messages import SystemMessage

def node1(state: State):

    SystemM = SystemMessage(
    content=(
        "You are a task classifier. "
        "Classify ONLY the user's latest request as either 'simple' or 'complex'. "
        "Do not answer the user's request.\n\n"

        "A simple task can be completed with a short, direct response "
        "and requires little reasoning or multiple steps.\n"
        "Examples:\n"
        "- 'What is 2 + 2?' -> simple\n"
        "- 'Translate hello into Spanish.' -> simple\n\n"

        "A complex task requires multiple steps, substantial reasoning, "
        "multiple constraints, or a detailed output.\n"
        "Examples:\n"
        "- 'Design a complete backend architecture for a medical application.' -> complex\n"
        "- 'Analyze this dataset, find trends, build a model, and explain the results.' -> complex\n\n"

        "Return the classification using the required structured output."
    )
)
    message = [SystemM, state["messages"][-1]]

    chat = llm("openai/gpt-oss-120b")
    structure_chat = chat.with_structured_output(Classification)
    result = structure_chat.invoke(message)

    return{ "category": result.category }